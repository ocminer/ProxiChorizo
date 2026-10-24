#!/usr/bin/env python3
"""Exercise packaged launcher arguments with a CPU-only executable stub."""
import json
import os
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
count = 0
with tempfile.TemporaryDirectory(prefix="proxichorizo-launchers-") as temp:
    base = pathlib.Path(temp)
    for launcher in ["hiveos/h-run.sh", "mmpos/mmpos-run.sh", "smos/smos-run.sh", "mmpos/mmp-launch.sh"]:
        for backend in ["cuda", "cl"]:
            for override in [False, True] if launcher.startswith("hiveos") else [False]:
                work = base / str(count)
                work.mkdir()
                shutil.copy2(ROOT / launcher, work / pathlib.Path(launcher).name)
                capture = work / "arguments.json"
                stub = '#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["PX_TEST_CAPTURE"],"w") as f:json.dump(sys.argv,f)\n'
                for name in ["proxichorizo", "proxichorizo-cl"]:
                    (work / name).write_text(stub)
                    (work / name).chmod(0o755)
                (work / "h-manifest.conf").write_text('CUSTOM_CONFIG_FILENAME="' + str(work / "unused.conf") + '"\nCUSTOM_LOG_BASENAME="' + str(work / "miner") + '"\n')
                (work / "mmp-external.conf").write_text("CUSTOM_API_PORT=4078\n")
                env = dict(os.environ)
                for name in ["SEED", "PASSWORD", "VARIANT", "THREADS", "ITERS", "MINER_GROUPS", "EXTRA_ARGS", "DEVICES"]:
                    env.pop(name, None)
                env.update(POOL_URL="http://example.invalid:8100", TOKEN="test", KEY_FILE="unused.key", MINER=backend, PX_TEST_CAPTURE=str(capture))
                if override:
                    env["MINER_GROUPS"] = "123"
                cmd = ["bash", str(work / pathlib.Path(launcher).name)]
                if launcher.endswith("mmp-launch.sh"):
                    cmd += ["--pool", "http://example.invalid:8100", "--user", "unit-test-credential", "--password", "unit-test-seed"]
                result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                argv = json.loads(capture.read_text())
                assert pathlib.Path(argv[0]).name == ("proxichorizo-cl" if backend == "cl" else "proxichorizo")
                assert "--groups" not in argv, "Bash GROUPS must not override GPU defaults"
                if override:
                    assert argv[argv.index("--blocks") + 1] == "123"
                else:
                    assert "--blocks" not in argv
                assert "unit-test-credential" not in result.stdout and "unit-test-seed" not in result.stdout
                count += 1
print("Launcher regression:", count, "CPU-only cases passed")
