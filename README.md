# gradio_stablediff_DINO

## Description

This repository contains a Gradio interface for the StableDiff model and the DINO model.
In one of the tabs, the user can input a text explaining what he wants to see, and he will get the image generated by the StableDiff model as an output.
In the other tab, the user can input an image and some text explaining what he wants to identify in the image, and he will get the image with the identified object highlighted as an output.

Models used:

- [StableDiff](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
- [DINO](https://huggingface.co/IDEA-Research/grounding-dino-base)

These two model are very large and require a lot of memory to run.
If you want to run this interface on your local machine, you will need a lot of memory (RAM) or a powerful GPU (NVDIA).


## Requirements

python >= 3.8

**Setting Up the Environment**

* Windows: `./setup_Windows.bat`
* Linux/macOS: `./setup_Linux.sh`

These scripts will install required dependencies, and build a virtual environment for you if you don't have one.


In order to install the required models, you will need to have the Hugging Face CLI installed, and you will need to be logged in.
Run the following command to login to the Hugging Face CLI:

```bash
$> huggingface-cli login
```

This will install the StableDiff and DINO models in the `models` directory.

## Running the Program

### CLI

1. Navigate to the `bin` directory: `cd bin`

2. Execute `python gradio_stable_dino.py [--help]` (use `python3` on Linux/macOS) in your terminal

    The `--help` flag displays available command-line arguments.

3. The Gradio interface will open in your browser automatically. It will use the localhost IP address and port 7860 by default.

## Author

Neetre
