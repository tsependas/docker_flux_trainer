from diffusers import FluxPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
dtype = torch.bfloat16  # попробуй float32 для начала

pipe = FluxPipeline.from_pretrained(
    "./workspace/flux1-dev",
    torch_dtype=dtype,
    local_files_only=True
)
pipe = pipe.to(device)
print(f"Model loaded on device: {device}")

prompt = "A cat holding a sign that says 'Hello World'"
out = pipe(
    prompt,
    guidance_scale=4.0,
    num_inference_steps=4,  # поставь 4, чтобы не ждать, потом увеличь
    output_type="pil"
)

image = out.images[0]
image.save("flux_output.png")
print("Изображение сохранено как flux_output.png")
