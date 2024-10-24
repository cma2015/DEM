import argparse
import os
import platform
import site
import shutil
from biodem import CollectFitLog


def cli_rm_ckpt():
    r"""Remove inferior models' checkpoints.
    """
    parser = argparse.ArgumentParser(description="Remove inferior models' checkpoints.")
    parser.add_argument("-p", "--path", type=str, help="Path to a directory containing fitting logs.", required=True)
    parser.add_argument("-r", "--report", type=str, help="Path to a directory to save the report.", required=False, default=None)
    args = parser.parse_args()
    ckpt_collector = CollectFitLog(args.path)
    ckpt_collector.remove_inferior_models()
    if args.report is not None:
        ckpt_collector.get_df_csv(args.report, overwrite_collected_log=True)


def cli_install_pregv():
    r"""Install pregv.
    """
    # parser = argparse.ArgumentParser(description="Install compiled pregv tool.")
    # Find the path where packages installed, like "$HOME/miniforge3/envs/env_name/lib/python3.12/site-packages"
    site_packages_path = site.getsitepackages()[0]
    # Define the path where the compiled tool will be copied to
    dst_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(site_packages_path))), "bin")
    print(f"Destination directory: {dst_dir}")
    if not os.path.exists(dst_dir):
        raise SystemExit(f"Destination directory does not exist.")

    # Get the current platform and architecture
    _platform = platform.system()
    _architecture = platform.machine()
    bname = "pregv"

    # Check the architecture
    if _architecture != 'x86_64':
        raise SystemExit(f"Unsupported architecture: {_architecture}, please compile {bname} manually.")

    # Check the platform and copy the tool
    if _platform == 'Linux':
        dst_path = os.path.join(dst_dir, bname)
        source_path = os.path.join(site_packages_path, "bin", "linux", bname)
        shutil.copyfile(source_path, dst_path)
        os.chmod(dst_path, 0o755)
        print(f"{bname} has been installed successfully: {dst_path}")
    elif _platform == 'Windows':
        dst_path = os.path.join(dst_dir, bname + ".exe")
        source_path = os.path.join(site_packages_path, "bin", "windows", bname + ".exe")
        shutil.copyfile(source_path, dst_path)
        os.chmod(dst_path, 0o755)
        print(f"{bname} has been installed successfully: {dst_path}")
    else:
        raise SystemExit(f"Unsupported platform: {_platform}, please compile {bname} manually.")

