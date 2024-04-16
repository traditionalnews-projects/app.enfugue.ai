import os
import subprocess
import streamlit as st

# This Python script serves as a one-and-done download, update, configure and run command for enfugue.
# It's able to download enfuge via conda or in portable form.

# Below is the configuration for enfugue. Most values are self-explanatory, but some others need some explanation.
# See below the configuration for explanation on these values.

config = """\
---
sandboxed: false                                # true = disable system management in UI
server:
    host: 0.0.0.0                               # listens on any connection
    port: [45554, 45555]                        # ports < 1024 require sudo
    domain: [app.enfugue.ai, null]              # this is a loopback domain, null = match request
    secure: [true, false]                       # enables SSL for first server, off for second
    # If you change the domain, you must provide your own certificates or disable SSL
    # key: /path/to/key.pem
    # cert: /path/to/cert.pem
    # chain: /path/to/chain.pem
    logging:
        file: ~/.cache/enfugue.log              # server logs (NOT diffusion logs)
        level: error                            # server only logs errors
    # cms:                                      # only configure when using a proxy
    #     path:
    #         root: http(s)://my-proxy-host.ngrok-or-other/
enfugue:
    noauth: true                                # authentication default
    queue: 4                                    # queue size default
    safe: true                                  # safety checker default
    model: v1-5-pruned.ckpt                     # default model (repo or file)
    inpainter: null                             # default inpainter (repo or file), null = sd15 base inpainter
    refiner: null                               # default refiner (repo or file), null = none
    refiner_start: 0.85                         # default start for refining when refiner set and no args (0.0-1.0)
    dtype: null                                 # set to float32 to disable half-precision, you probably dont want to
    engine:
        logging:
            file: ~/.cache/enfugue-engine.log   # diffusion logs (shown in UI)
            level: debug                        # logs everything, helpful for debugging
        root: ~/.cache/enfugue                  # root engine directory, images save in /images
        cache: ~/.cache/enfugue/cache           # diffusers cache, controlnets, VAE
        checkpoint: ~/.cache/enfugue/checkpoint # checkpoints only
        lora: ~/.cache/enfugue/lora             # lora only
        lycoris: ~/.cache/enfugue/lycoris       # lycoris only
        inversion: ~/.cache/enfugue/inversion   # textual inversion only
        motion: ~/.cache/enfugue/motion         # motion modules only
        other: ~/.cache/enfugue/other           # other AI models (upscalers, preprocessors, etc.)
    pipeline:
        switch: "offload"                   # See comment above
        inpainter: true                     # See comment above
        cache: null                         # See comment above
        sequential: false                   # See comment above
"""

with open("config.yml", "w") as f:
    f.write(config)

# -----------------------
# enfugue.pipeline.switch
# -----------------------
# 'switch' determines how to swap between pipelines when required, like going from inpainting to non-inpainting or loading a refiner.
# The default behavior, 'offload,' sends unneeded pipelines to the CPU and promotes active pipelines to the GPU when requested.
# This usually provides the best balance of speed and memory usage, but can result in heavy overhead on some systems.
#
# If this proves too much, or you wish to minimize memory usage, set this to 'unload,'which willalways completely unload a pipeline 
# and free memory before a different pipeline is used.
#
# If you set this to 'null,' _all models will remain in memory_. This is by far thefastest but consumes the most memory, this is only
# suitable for enterprise GPUs.
#
# --------------------------
# enfugue.pipeline.inpainter
# --------------------------
# 'inpainter' determines how to inpaint when no inpainter is specified.
#
# When the value is 'true', and the user is using a stable diffusion 1.5 model for their base model, enfugue will look for another
# checkpoint with the same name but the suffix `-inpainting`, and when one is not found, it will create one using the model merger.
# Fine-tuned inpainting checkpoints perform significantly better at the task, however they are roughly equivalent in size to the
# main model, effectively doubling storage required.
#
# When the value is 'null' or 'false', Enfugue will still search for a fine-tuned inpainting model, but will not create one if it does not exist.
# Instead, enfugue will use 4-dim inpainting, which in 1.5 is less effective.
# SDXL does not have a fine-tuned inpainting model (yet,) so this procedure does not apply, and 4-dim inpainting is always used.
#
# ----------------------
# enfugue.pipeline.cache
# ----------------------
# 'cache' determines when to create diffusers caches. A diffusers cache will always load faster than a checkpoint, but is once again
# approximately the same size as the checkpoint, so this will also effectively double storage size.
#
# When the value is 'null' or 'false', diffusers caches will _only_ be made for TensorRT pipelines, as it is required. This is the default value.
#
# When the value is 'xl', enfugue will cache XL checkpoints. These load _significantly_ faster than when loading from file, between
# 2 and 3 times as quickly. You may wish to consider using this setting to speed up changing between XL checkpoints.
#
# When the value is 'true', diffusers caches will be created for all pipelines. This is not recommended as it only provides marginal
# speed advantages for 1.5 models.
#
# ---------------------------
# enfugue.pipeline.sequential
# ---------------------------
# 'sequential' enables sequential onloading and offloading of AI models.
#
# When the value is 'true', AI models will only ever be loaded to the GPU when they are needed.
# At all other times, they will be in normal memory, waiting for the next time they are requested, at which time they will be loaded
# to the GPU, and afterward unloaded.
#
# These operations take time, so this is only recommended to enable ifyou are experiencing issues with out-of-memory errors.
#
# -- end of configuration --
# -- start functions --

def usage():
    """Print out the help message."""
    print("""\
USAGE: %s [OPTIONS]
Options:
  --help                   Display this help message.
  --conda / --portable     Automatically set installation type (do not prompt.)
  --update / --no-update   Automatically apply or skip updates (do not prompt.)
  --mmpose / --no-mmpose   Automatically install or skip installing MMPose (do not prompt.)
""" % os.path.basename(sys.argv[0]))

def compare_versions(v1, v2):
    """Compare version strings."""
    if v1 == v2:
        return 0
    ver1 = list(map(int, v1.split(".")))
    ver2 = list(map(int, v2.split(".")))
    for i in range(max(len(ver1), len(ver2))):
        a = ver1[i] if i < len(ver1) else 0
        b = ver2[i] if i < len(ver2) else 0
        if a > b:
            return 1
        elif a < b:
            return -1
    return 0

def compare_prompt_update(v1, v2, install_update=False):
    """Compare versions and prompt to download when relevant."""
    compare = compare_versions(v1, v2)
    if compare < 0:
        if install_update:
            return True
        else:
            print(f"Version {v2} of enfugue is available, you have version {v1} installed. Download update? [Yes]: ")
            answer = input().lower()
            return answer[0] in ["y", "t", "1"]

def download_portable():
    """Download and extract the latest portable version."""
    import requests
    import tarfile
    import shutil

    url = "https://api.github.com/repos/painebenjamin/app.enfugue.ai/releases/latest"
    response = requests.get(url)
    data = response.json()
    assets = data["assets"]
    for asset in assets:
        if "manylinux" in asset["name"]:
            url = asset["browser_download_url"]
            filename = url.split("/")[-1]
            response = requests.get(url, stream=True)
            with open(filename, "wb") as f:
                shutil.copyfileobj(response.raw, f)
            tar = tarfile.open(filename)
            tar.extractall()
            tar.close()
            os.remove(filename)
            break

# -- end functions --
# -- start script --

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--conda", action="store_true")
parser.add_argument("--portable", action="store_true")
parser.add_argument("--mmpose", action="store_true")
parser.add_argument("--no-mmpose", action="store_false", dest="mmpose")
parser.add_argument("--update", action="store_true")
parser.add_argument("--no-update", action="store_false", dest="update")
args = parser.parse_args()

# Set default options, then iterate through command line arguments and set variables.
install_type = ""
install_mmpose = None
install_update = None

if args.conda:
    install_type = "conda"
elif args.portable:
    install_type = "portable"

if args.mmpose:
    install_mmpose = True
elif args.no_mmpose:
    install_mmpose = False

if args.update:
    install_update = True
elif args.no_update:
    install_update = False

# Set portable directory and paths
portable_dir = os.path.join(os.getcwd(), "enfugue-server")
os.environ["PATH"] = os.environ["PATH"] + os.pathsep + os.getcwd() + os.pathsep + portable_dir + os.pathsep

# Gather some variables from the current environment.
conda = shutil.which("conda")

# Make sure conda can be executed.
if conda and not args.portable:
    if subprocess.run(["conda", "env", "list"], stdout=subprocess.DEVNULL).returncode == 0:
        print("Found enfugue environment, activating.")
        subprocess.run(["source", "activate", "enfugue"])
        enfugue = shutil.which("enfugue")
    else:
        enfugue = None
else:
    enfugue = None

enfugue_server = shutil.which("enfugue-server")

# Change variables if forcing portable/conda
if install_type == "conda":
    enfugue_server = None
elif install_type == "portable":
    enfugue = None

# These will be populated later if relevant.
enfugue_installed_pip_version = None
enfugue_available_pip_version = None
enfugue_installed_portable_version = None
enfugue_available_portable_version = None

# Get the current python executable
python = shutil.which("python3")

# Check if either of the above tactics found enfugue. If so, and it's not disabled,check for updates.
if enfugue or enfugue_server:
    if python and install_update is None:
        # Get installed version from pip
        enfugue_installed_pip_version = subprocess.run([python, "-m", "pip", "freeze"], stdout=subprocess.PIPE).stdout.decode().strip().split("\n")
        enfugue_installed_pip_version = [line.split("==")[-1] for line in enfugue_installed_pip_version if "enfugue" in line]
    if enfugue and enfugue_installed_pip_version and install_update is None:
        # Get available versions from pip
        enfugue_available_pip_version = subprocess.run([python, "-m", "pip", "install", "enfugue==", "--dry-run"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr.decode().strip().split("\n")
        enfugue_available_pip_version = [line.split(" ")[-1] for line in enfugue_available_pip_version if "enfugue" in line]
        enfugue_available_pip_version = [version.split("==")[-1] for version in enfugue_available_pip_version]
    if enfugue_server and install_update is None:
        # Get versions
        enfugue_available_portable_version = subprocess.run(["curl", "-s", "https://api.github.com/repos/painebenjamin/app.enfugue.ai/releases/latest"], stdout=subprocess.PIPE).stdout.decode().strip()
        enfugue_available_portable_version = enfugue_available_portable_version.split("tag_name": "")[1].split(",")[0].replace('"', "")

        # Check if the installed version is outdated
        if compare_prompt_update(enfugue_installed_portable_version, enfugue_available_portable_version):
            print("Downloading the latest version...")
            download_portable()
            enfugue_installed_portable_version = enfugue_available_portable_version
            print(f"Successfully downloaded version {enfugue_installed_portable_version}.")

# Display the current version
if enfugue_installed_pip_version:
    print(f"Enfugue version (pip): {enfugue_installed_pip_version[0]}")
if enfugue_installed_portable_version:
    print(f"Enfugue version (portable): {enfugue_installed_portable_version}")

# Create the enfugue.yml file
if enfugue:
    with open("enfugue.yml", "w") as f:
        f.write(config)

# Run enfugue
if enfugue:
    subprocess.run([enfugue, "run"])
elif enfugue_server:
    subprocess.run([enfugue_server])
else:
    print("Enfugue not found.")
