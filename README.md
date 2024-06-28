# MISTOLINE

MistoLine is a versatile and robust SDXL-ControlNet model developed by TheMistoAI that can adapt to any type of line art input, demonstrating high accuracy and excellent stability

- It can generate high-quality images (with a short side greater than 1024px) based on user-provided line art of various types, including hand-drawn sketches, different ControlNet line preprocessors, and model-generated outlines.

## KEY FEATURES

- Employing a novel line preprocessing algorithm called Anyline, which accurately extracts object edges, image details, and textual content from most images
- Retraining the ControlNet model based on the Unet of stabilityai/stable-diffusion-xl-base-1.0, along with innovations in large model training engineering
- Showcasing superior performance across different types of line art inputs, surpassing existing ControlNet models in terms of detail restoration, prompt alignment, and stability, particularly in more complex scenarios
- Eliminating the need to select different ControlNet models for different line preprocessors, as it exhibits strong generalization capabilities across diverse line art conditions


---

## YT Recording

<https://youtu.be/xdBM_qEGdF8>

---

## REFERENCES

1. <https://huggingface.co/TheMistoAI/MistoLine>
2. <https://github.com/TheMistoAI/MistoLine>

---