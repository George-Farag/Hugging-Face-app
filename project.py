import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import gradio as gr

# Load models
model = AutoModelForCausalLM.from_pretrained(
    "maya-research/maya1",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")

# Main generation function
def generate_voice(description, text):
    prompt = f'<description="{description}"> {text}'
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.4,
            top_p=0.9,
            do_sample=True
        )

    generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
    snac_tokens = [t.item() for t in generated_ids if 128266 <= t <= 156937]

    frames = len(snac_tokens) // 7
    codes = [[], [], []]
    for i in range(frames):
        s = snac_tokens[i*7:(i+1)*7]
        codes[0].append((s[0]-128266) % 4096)
        codes[1].extend([(s[1]-128266) % 4096, (s[4]-128266) % 4096])
        codes[2].extend([(s[2]-128266) % 4096, (s[3]-128266) % 4096,
                         (s[5]-128266) % 4096, (s[6]-128266) % 4096])

    codes_tensor = [torch.tensor(c, dtype=torch.long, device="cuda").unsqueeze(0) for c in codes]
    with torch.inference_mode():
        audio = snac_model.decoder(snac_model.quantizer.from_codes(codes_tensor))[0, 0].cpu().numpy()

    out_path = "output.wav"
    sf.write(out_path, audio, 24000)
    return out_path

# Gradio interface — no preset text, fully user-controlled
demo = gr.Interface(
    fn=generate_voice,
    inputs=[
        gr.Textbox(label="Voice Description (e.g., calm female voice with British accent)"),
        gr.Textbox(label="Text to Speak (type anything you want)")
    ],
    outputs=gr.Audio(label="Generated Speech"),
    title="🎙️ Maya1 Voice Generator",
    description="Generate expressive emotional speech using the open-source Maya1 + SNAC pipeline."
)

if __name__ == "__main__":
    demo.launch()
