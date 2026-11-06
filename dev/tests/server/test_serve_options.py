import io
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from install import launcher
from server import images, serve_options
from server import server as api
from server.chat_format import REASONING_EFFORTS
from server.origins import ANY_ORIGIN

ROOT = Path(__file__).resolve().parents[3]
# Each parser, with the arguments it requires.
SERVER_ARGS = ["model", "--model", "owner/repo"]
LAUNCHER_ARGS = ["serve", "--model", "owner/repo"]
PARSERS = ((api.parse_args, SERVER_ARGS), (launcher.parse_args, LAUNCHER_ARGS))


def values(flag, accepted, refused, repeatable=False):
    """Arguments that give `flag` each accepted value, with the value both
    parsers parse it to, and arguments that give it each refused value."""
    return (
        [
            ([f"{flag}={text}"], [parsed] if repeatable else parsed)
            for text, parsed in accepted.items()
        ],
        [[f"{flag}={text}"] for text in refused],
    )


# Up to the native window and past it, where the engine serves the context
# with YaRN, to the 1010000 tokens Qwen validates; K is 1024 tokens and M a
# million.
ACCEPTED_CONTEXTS = {
    "auto": None,
    "100K": 102400,
    "256k": 262144,
    "262144": 262144,
    "1": 1,
    "262145": 262145,
    "512K": 524288,
    "1m": 1000000,
    "1000000": 1000000,
    "1010000": 1010000,
}
# Past 1010000 tokens (987K is 1010688), or no count of tokens.
REFUSED_CONTEXTS = ("0", "-1", "1010001", "987K", "2M", "1.5M", "M", "bad")
# Every shared option: accepted arguments with the value they parse to, and
# refused arguments.
OPTIONS = {
    "--host": values(
        "--host", {"0.0.0.0": "0.0.0.0", "mymac.local": "mymac.local"}, ()
    ),
    "--served-model-name": values(
        "--served-model-name",
        {"local": "local", "-local": "-local", "org/model": "org/model"},
        ("", "has space", "a?b", "a#b", "a%b", "a\\b", "/a", "a//b", "a/../b"),
        repeatable=True,
    ),
    "--announce-served-name": (
        [(["--announce-served-name", "--served-model-name=local"], True)],
        [["--announce-served-name"]],
    ),
    "--default-reasoning-effort": values(
        "--default-reasoning-effort",
        {effort: effort for effort in REASONING_EFFORTS},
        ("", "turbo", "XHIGH"),
    ),
    "--thinking-token-budget": values(
        "--thinking-token-budget", {"0": 0, "4096": 4096}, ("-1", "1.5", "many")
    ),
    "--thinking-end-text": values(
        "--thinking-end-text",
        {"</think>": "</think>", "Time is up.\n</think>": "Time is up.\n</think>"},
        (),
    ),
    "--kv-format": values("--kv-format", {"int8": "int8", "bf16": "bf16"}, ("fp16",)),
    "--state-format": values(
        "--state-format", {"fp32": "fp32", "bf16": "bf16"}, ("fp16", "int8")
    ),
    "--prefill-mode": values(
        "--prefill-mode", {"bounded": "bounded", "full": "full"}, ("fast", "")
    ),
    "--prefill-bound-ms": values(
        "--prefill-bound-ms",
        {"1": 1, "3000": 3000, "600000": 600000},
        ("0", "-5", "2s", "600001", "1.5"),
    ),
    "--prefill-bound-scope": values(
        "--prefill-bound-scope",
        {"buffer": "buffer", "command": "command"},
        ("layer", "", "Command"),
    ),
    "--disable-ane": ([(["--disable-ane"], True)], []),
    "--ane-ffn-share": (
        values("--ane-ffn-share", {"0": 0.0, "0.35": 0.35, "0.999": 0.999}, ())[0],
        [
            *values("--ane-ffn-share", {}, ("1", "-0.1", "nan", "inf", "half"))[1],
        ],
    ),
    "--ane-ffn-minimum-rows": (
        [(["--ane-ffn-minimum-rows=520", "--ane-ffn-share=0.35"], 520)],
        [
            ["--ane-ffn-minimum-rows=520"],
            *values("--ane-ffn-minimum-rows", {}, ("0", "-1", "1.5", "rows"))[1],
        ],
    ),
    "--verify-rows": values(
        "--verify-rows",
        {"auto": "auto", "all": "all"},
        ("", "budget", "every", "ALL"),
    ),
    "--ple-memory": values(
        "--ple-memory",
        {
            "auto": None,
            "off": "off",
            "OFF": "off",
            "1M": 1024**2,
            "2G": 2 * 1024**3,
            "0.75G": 768 * 1024**2,
        },
        ("0", "bad", "-2G", ".5G"),
    ),
    "--expert-memory": values(
        "--expert-memory",
        {
            "auto": None,
            "off": "off",
            "OFF": "off",
            "512M": 512 * 1024**2,
            "16G": 16 * 1024**3,
        },
        ("0", "bad", "-2G"),
    ),
    "--sparse-kv-memory": values(
        "--sparse-kv-memory",
        {"auto": None, "AUTO": None, "1200M": 1200 * 1024**2},
        ("0", "off", "bad", "-2G"),
    ),
    "--max-memory": values(
        "--max-memory",
        {
            "auto": None,
            "28G": 28 * 1024**3,
            "1GiB": 1024**3,
            "512mb": 512 * 1024**2,
            "1073741824": 1024**3,
            "10.4G": 11166914969,
            "2.5g": 5 * 512 * 1024**2,
        },
        ("0", "-1G", "bad", "G", str(2**64), "1.5", "2.G", "1e3G", "0.0000000001K"),
    ),
    "--idle-release": values(
        "--idle-release",
        {"600": 600.0, "90s": 90.0, "30M": 1800.0, "1.5h": 5400.0, "off": math.inf},
        ("0", "-1m", "inf", "nan", "10x", "m", "never"),
    ),
    "--max-cache-disk": values(
        "--max-cache-disk",
        {"auto": None, "0": 0, "5G": 5 * 1024**3},
        ("-1", "0G", "5X", "nan"),
    ),
    "--max-cache-disk-writes": (
        [
            (["--max-cache-disk-writes=0"], 0),
            (["--max-cache-disk-writes=200G"], 200 * 1024**3),
        ],
        [
            ["--max-cache-disk-writes=200G", "--max-cache-disk=0"],
            ["--max-cache-disk-writes=-1"],
            ["--max-cache-disk-writes=5X"],
            ["--max-cache-disk-writes=auto"],
        ],
    ),
    "--persistent-cache": (
        [
            (["--persistent-cache"], True),
            (["--persistent-cache", "--max-cache-disk=5G"], True),
        ],
        [["--persistent-cache", "--max-cache-disk=0"]],
    ),
    "--cache-dir": (
        [
            (
                ["--cache-dir=/srv/cache", "--persistent-cache", "--max-cache-disk=5G"],
                Path("/srv/cache"),
            ),
            (
                ["--cache-dir=~/cache", "--persistent-cache", "--max-cache-disk=5G"],
                Path.home() / "cache",
            ),
        ],
        [
            ["--cache-dir=/srv/cache"],
            ["--cache-dir=", "--persistent-cache", "--max-cache-disk=5G"],
        ],
    ),
    "--max-context": values("--max-context", ACCEPTED_CONTEXTS, REFUSED_CONTEXTS),
    "--decode-share": values(
        "--decode-share",
        {"0": 0.0, "0.25": 0.25, "2": 2.0},
        ("-0.5", "inf", "nan", "half"),
    ),
    "--contended-prefill-rows": values(
        "--contended-prefill-rows",
        {"64": 64, "256": 256, "2048": 2048},
        ("0", "32", "63", "96", "4096", "bad"),
    ),
    "--allowed-host": values(
        "--allowed-host", {"proxy.example": "proxy.example"}, (), repeatable=True
    ),
    "--allowed-origin": values(
        "--allowed-origin",
        {
            "tauri://localhost": ("tauri", "localhost", None),
            "http://[::1]:3000": ("http", "::1", 3000),
            "*": ANY_ORIGIN,
        },
        ("http://localhost/app", "tauri://*", "localhost"),
        repeatable=True,
    ),
    "--max-request-size": values(
        "--max-request-size",
        {"256M": 256 * 1024**2, "1G": 1024**3},
        ("auto", "0", "-1G", "bad", str(2**64)),
    ),
    "--max-image-pixels": values(
        "--max-image-pixels",
        {str(images.MIN_PIXELS): images.MIN_PIXELS, "1048576": 1048576},
        (str(images.MIN_PIXELS - 1), str(images.MAX_PIXELS + 1), "-1", "many"),
    ),
    "--request-timeout": values(
        "--request-timeout",
        {"3600": 3600.0, "0.5": 0.5, "90s": 90.0, "30M": 1800.0, "1.5h": 5400.0},
        ("0", "-1", "-1m", "inf", "nan", "10x", "m", "off", "soon"),
    ),
    "--queue-size": values(
        "--queue-size", {"64": 64, "1": 1}, ("0", "-1", "1.5", "many")
    ),
    "--api-key": values(
        "--api-key",
        {"secret-key": "secret-key"},
        ("", "two words", "key\n", "非ASCII"),
    ),
    "--allow-idle-sleep": ([(["--allow-idle-sleep"], True)], []),
    "--no-webui": ([(["--no-webui"], True)], []),
}


def dest(flag):
    return flag.removeprefix("--").replace("-", "_")


def native_command_lines():
    """(name, serve options, serve-native arguments after the model directory)
    of each line of the command lines both sides read."""
    golden = api.ROOT / "dev/tests/engine/native_command_golden.txt"
    for line in golden.read_text().splitlines():
        if line and not line.startswith("#"):
            name, options, arguments = line.split("|")
            yield name.strip(), options.split(), arguments.split()


class ServeOptionsTests(unittest.TestCase):
    def setUp(self):
        # The parsers' defaults ignore the caller's Splash settings.
        self.enterContext(mock.patch.dict(os.environ))
        for name in (
            "SPLASH_API_KEY",
            "SPLASH_DEFAULT_REASONING_EFFORT",
            "SPLASH_THINKING_TOKEN_BUDGET",
            "SPLASH_THINKING_END_TEXT",
        ):
            os.environ.pop(name, None)

    def refuse(self, parse, arguments):
        with (
            mock.patch("sys.stderr", io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            parse(arguments)
        self.assertEqual(raised.exception.code, 2)
        return stderr.getvalue()

    def test_both_parsers_accept_and_reject_the_same_values(self):
        self.assertEqual(
            set(OPTIONS), {option.flag for option in serve_options.SERVE_OPTIONS}
        )
        defaults = [vars(parse(required)) for parse, required in PARSERS]
        for option in serve_options.SERVE_OPTIONS:
            self.assertEqual(
                defaults[0][option.dest], defaults[1][option.dest], option.flag
            )
        for flag, (accepted, refused) in OPTIONS.items():
            for (parse, required), (arguments, parsed) in (
                (parser, case) for parser in PARSERS for case in accepted
            ):
                with self.subTest(parser=parse.__module__, arguments=arguments):
                    self.assertEqual(
                        getattr(parse([*required, *arguments]), dest(flag)), parsed
                    )
            for (parse, required), arguments in (
                (parser, case) for parser in PARSERS for case in refused
            ):
                with self.subTest(parser=parse.__module__, arguments=arguments):
                    self.refuse(parse, [*required, *arguments])

    def test_both_parsers_check_defaults_from_the_environment(self):
        for name, value, parsed, flag in (
            (
                "SPLASH_DEFAULT_REASONING_EFFORT",
                "low",
                "low",
                "--default-reasoning-effort",
            ),
            ("SPLASH_THINKING_TOKEN_BUDGET", "4096", 4096, "--thinking-token-budget"),
            ("SPLASH_API_KEY", "environment-key", "environment-key", "--api-key"),
        ):
            for parse, required in PARSERS:
                with (
                    self.subTest(parser=parse.__module__, name=name),
                    mock.patch.dict(os.environ, {name: value}),
                ):
                    self.assertEqual(getattr(parse(required), dest(flag)), parsed)
                    # The command line outranks the environment.
                    explicit = OPTIONS[flag][0][0]
                    self.assertEqual(
                        getattr(parse([*required, *explicit[0]]), dest(flag)),
                        explicit[1],
                    )
                with (
                    self.subTest(parser=parse.__module__, name=name, valid=False),
                    mock.patch.dict(os.environ, {name: "two words"}),
                ):
                    self.assertIn(flag, self.refuse(parse, required))

    def test_both_parsers_take_the_thinking_end_text_from_the_environment(self):
        # Any text parses; the server checks it with the model's tokenizer.
        text = "Time is up.\n</think>"
        for parse, required in PARSERS:
            with (
                self.subTest(parser=parse.__module__),
                mock.patch.dict(os.environ, {"SPLASH_THINKING_END_TEXT": text}),
            ):
                self.assertEqual(parse(required).thinking_end_text, text)
                # The command line outranks the environment.
                given = parse([*required, "--thinking-end-text=</think>"])
                self.assertEqual(given.thinking_end_text, "</think>")

    def test_each_parse_has_its_own_lists(self):
        for parse, required in PARSERS:
            for option in serve_options.SERVE_OPTIONS:
                if option.options.get("action") != "append":
                    continue
                with self.subTest(parser=parse.__module__, flag=option.flag):
                    getattr(parse(required), option.dest).append("changed")
                    self.assertEqual(getattr(parse(required), option.dest), [])

    def test_launcher_forwards_exactly_what_the_server_parses(self):
        launched = [
            *(
                argument
                for accepted, _ in OPTIONS.values()
                for arguments, _ in accepted[-1:]
                for argument in arguments
            ),
            # A repeatable option keeps every value, in order.
            "--served-model-name=stable",
            "--allowed-host=proxy.local",
            "--allowed-origin=tauri://localhost",
        ]
        args = launcher.parse_args([*LAUNCHER_ARGS, *launched])
        argv = serve_options.serve_argv(args)
        environment = serve_options.serve_environment(args)
        # Every option but the key travels on the command line; the key only
        # in the environment.
        self.assertEqual(
            {argument.partition("=")[0] for argument in argv},
            {
                option.flag
                for option in serve_options.SERVE_OPTIONS
                if not option.secret
            },
        )
        self.assertFalse(any("secret-key" in argument for argument in argv))
        self.assertEqual(environment, {"SPLASH_API_KEY": "secret-key"})
        with mock.patch.dict(os.environ, environment):
            served = api.parse_args([*SERVER_ARGS, *argv])
        for option in serve_options.SERVE_OPTIONS:
            self.assertEqual(
                getattr(served, option.dest), getattr(args, option.dest), option.flag
            )
        # Defaults stay with the server, which reads them itself.
        defaults = launcher.parse_args(LAUNCHER_ARGS)
        self.assertEqual(serve_options.serve_argv(defaults), [])
        self.assertEqual(serve_options.serve_environment(defaults), {})

    def test_the_launcher_imports_only_the_standard_library(self):
        """The launcher runs before the environment it sets up exists: it,
        and the serve options it reads, import nothing beside the standard
        library and Splash's own packages."""
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "import install.launcher; "
                "print(sorted({name.partition('.')[0] for name in sys.modules} "
                "- set(sys.stdlib_module_names)))",
                str(ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "['__main__', 'install', 'server']")

    def test_server_requires_explicit_model_and_paths(self):
        model = "community/custom-splash"
        package = api.ROOT / "install/models" / model
        required = [str(package), "--model", model]
        for arguments in (
            [],
            required[:-2],
            [*required[:-1], "qwen3.8-27b"],
            [*required[:-1], "Qwen3.8-27B"],
            [*required[:-1], "https://huggingface.co/community/model"],
            [*required[:-1], "community/../model"],
        ):
            with (
                self.subTest(arguments=arguments),
                mock.patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                api.parse_args(arguments)
        args = api.parse_args(required)
        self.assertEqual(Path(args.model_root), package)
        self.assertIsNone(args.request_timeout)
        self.assertEqual(args.model, model)
        self.assertEqual(Path(args.binary).name, "splash")
        with mock.patch("sys.stderr"):
            for value in ("-1", "65536"):
                with self.assertRaises(SystemExit):
                    api.parse_args([*required, "--port", value])

    def test_native_command_passes_each_serve_option(self):
        model = "community/custom-splash"
        package = api.ROOT / "install/models" / model
        required = [str(package), "--model", model]
        args = api.parse_args(required)
        # The engine's command line for each option, which the engine reads as
        # the server means it (native_arguments_test.cpp). The engine keeps
        # its own default of an option the server does not pass, and the
        # per-image patch limit is the most patches a resized image within
        # the pixel cap can have, a whole number of merge units. A relaunch's
        # line follows a first engine that served 61433 tokens.
        for name, options, arguments in native_command_lines():
            with self.subTest(name=name):
                self.assertEqual(
                    api._native_command(
                        api.parse_args([*required, *options]),
                        args.model_root,
                        61433 if name.startswith("relaunch") else None,
                    ),
                    [args.binary, "serve-native", args.model_root, *arguments],
                )

        def native(parsed):
            return api._native_command(parsed, parsed.model_root)

        # The engine resolves the contended floor's GPU-family default itself;
        # a given floor reaches it as given.
        self.assertNotIn("--contended-prefill-rows", native(args))
        rows_args = api.parse_args([*required, "--contended-prefill-rows=128"])
        self.assertEqual(native(rows_args)[-2:], ["--contended-prefill-rows", "128"])

        # Prefill commands stay within 2 s unless the mode or bound says
        # otherwise; the engine applies the default itself.
        self.assertEqual(args.prefill_mode, "bounded")
        self.assertIsNone(args.prefill_bound_ms)
        self.assertNotIn("--prefill-mode", native(args))
        self.assertNotIn("--prefill-bound-ms", native(args))
        full_args = api.parse_args([*required, "--prefill-mode", "full"])
        self.assertEqual(native(full_args)[-2:], ["--prefill-mode", "full"])
        bound_args = api.parse_args([*required, "--prefill-bound-ms", "3000"])
        self.assertEqual(native(bound_args)[-2:], ["--prefill-bound-ms", "3000"])
        self.assertNotIn(
            "--prefill-bound-ms",
            native(
                api.parse_args(
                    [*required, "--prefill-mode", "full", "--prefill-bound-ms", "3000"]
                )
            ),
        )
        with mock.patch.dict(os.environ, {"SPLASH_PREFILL_BOUND_MS": "1500"}):
            self.assertEqual(api.parse_args(required).prefill_bound_ms, 1500)
        # The bound holds each command buffer unless told to hold whole
        # commands; full mode has no bound to scope.
        self.assertIsNone(args.prefill_bound_scope)
        self.assertNotIn("--prefill-bound-scope", native(args))
        self.assertNotIn(
            "--prefill-bound-scope",
            native(api.parse_args([*required, "--prefill-bound-scope", "buffer"])),
        )
        command_args = api.parse_args([*required, "--prefill-bound-scope", "command"])
        self.assertEqual(
            native(command_args)[-2:], ["--prefill-bound-scope", "command"]
        )
        self.assertNotIn(
            "--prefill-bound-scope",
            native(
                api.parse_args(
                    [
                        *required,
                        "--prefill-mode",
                        "full",
                        "--prefill-bound-scope",
                        "command",
                    ]
                )
            ),
        )
        with mock.patch.dict(os.environ, {"SPLASH_PREFILL_BOUND_SCOPE": "command"}):
            self.assertEqual(api.parse_args(required).prefill_bound_scope, "command")
        for invalid in (
            ["--prefill-mode", "fast"],
            ["--prefill-bound-ms", "0"],
            ["--prefill-bound-ms", "-5"],
            ["--prefill-bound-ms", "2s"],
            ["--prefill-bound-ms", "600001"],
            ["--prefill-bound-scope", "layer"],
        ):
            with (
                self.subTest(invalid=invalid),
                mock.patch("sys.stderr", io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                api.parse_args([*required, *invalid])

    def test_display_stall_bounds_whole_commands_then_halves_to_the_floor(self):
        package = api.ROOT / "install/models/community/custom-splash"
        required = [str(package), "--model", "community/custom-splash"]

        def stall_policy(argv):
            parsed = api.parse_args(argv)
            return api._shorter_prefill_after_display_stall(parsed, parsed.model_root)

        stall = api.engine_runtime.EngineUnhealthy(
            "Metal backend is unhealthy: Metal command 117 failed (sparse event 0): "
            "Impacting Interactivity "
            "(0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)"
        )

        def bounds(policy, aborts):
            """(scope, bound) of each relaunch, None where the policy gives up."""
            seen = []
            for _ in range(aborts):
                command = policy(stall)
                if command is None:
                    seen.append(None)
                    continue
                scope = (
                    command[command.index("--prefill-bound-scope") + 1]
                    if "--prefill-bound-scope" in command
                    else "buffer"
                )
                bound = (
                    command[command.index("--prefill-bound-ms") + 1]
                    if "--prefill-bound-ms" in command
                    else "2000"
                )
                seen.append((scope, bound))
            return seen

        with mock.patch.object(api, "print_status") as status:
            policy = stall_policy(required)
            self.assertIsNone(policy(api.engine_runtime.EngineUnhealthy("GPU is gone")))
            self.assertEqual(
                bounds(policy, 5),
                [
                    ("command", "2000"),
                    ("command", "1000"),
                    ("command", "500"),
                    None,
                    None,
                ],
            )
            self.assertEqual(status.call_count, 3)
            # Whole commands from the start: the first abort halves already.
            command_policy = stall_policy(
                [*required, "--prefill-bound-scope", "command"]
            )
            self.assertEqual(
                bounds(command_policy, 3),
                [("command", "1000"), ("command", "500"), None],
            )
            full = stall_policy([*required, "--prefill-mode", "full"])
            self.assertIsNone(full(stall))

    def test_a_relaunch_asks_for_the_first_context_as_relaunch_context(self):
        """A relaunch after --max-context asks for the first engine's context
        as an automatic one does, past the native window too, and serves
        what its plan holds of it (it required --max-context again). It keeps
        --max-context, so that it is planned as the first engine was."""
        model = "community/custom-splash"
        required = [str(api.ROOT / "install/models" / model), "--model", model]
        args = api.parse_args([*required, "--max-context", "1000000"])
        self.assertEqual(
            api._native_command(args, args.model_root, 1_000_000),
            [
                args.binary,
                "serve-native",
                args.model_root,
                "1000000",
                "auto",
                "--relaunch-context",
                "1000000",
            ],
        )


if __name__ == "__main__":
    unittest.main()
