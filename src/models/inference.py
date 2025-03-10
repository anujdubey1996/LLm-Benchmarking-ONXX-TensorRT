import torch
import time
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load configuration file
with open("../configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
MAX_TOKENS = config["model"]["max_new_tokens"]

class LLMInference:
    def __init__(self):
        """Initialize LLM model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def generate(self, prompt: str):
        """Run inference and return response + execution time."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Measure inference time
        start_time = time.time()
        outputs = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        end_time = time.time()

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "response": response,
            "inference_time": round(end_time - start_time, 2)
        }

# Run test inference
if __name__ == "__main__":
    inference = LLMInference()
    test_prompt = "Explain how transformers work in NLP."
    result = inference.generate(test_prompt)
    print(f"\nGenerated Text:\n{result['response']}")
    print(f"Inference Time: {result['inference_time']} seconds")
