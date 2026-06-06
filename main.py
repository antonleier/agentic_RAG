# pip install weave wandb
# wandb login

import asyncio
import random
import weave

weave.init("vejaleier-lss/weavehacks-rag")  # 🐝 same project as before

# ---- A fake "model" with a tunable quality knob -------------------------
# In a real system, `quality` would be your prompt, retrieval settings,
# fine-tuning, etc. Here it's just a number from 0.0 to 1.0 that makes
# the model "answer correctly" more often as it goes up.
class FakeRagModel(weave.Model):
    quality: float = 0.3  # starts mediocre

    @weave.op
    def predict(self, question: str, answer: str) -> str:
        # With probability = quality, it "knows" the right answer.
        # Otherwise it returns a wrong/garbage answer.
        if random.random() < self.quality:
            return answer  # got it right
        return "I'm not sure."  # got it wrong


# ---- A scorer: did the model's output match the expected answer? --------
@weave.op
def exact_match(answer: str, output: str) -> dict:
    return {"correct": output.strip().lower() == answer.strip().lower()}


# ---- A tiny fake dataset ------------------------------------------------
dataset = [
    {"question": "What library are we using?", "answer": "Weave"},
    {"question": "What does retrieve do?", "answer": "Fetches documents"},
    {"question": "What is the demo about?", "answer": "Self-improving RAG"},
    {"question": "Who hosts WeaveHacks?", "answer": "Weights & Biases"},
    {"question": "What climbs over time?", "answer": "The eval score"},
]


# ---- The "self-improvement" loop ----------------------------------------
async def main():
    model = FakeRagModel()
    evaluation = weave.Evaluation(
        name="self-improvement-loop",
        dataset=dataset,
        scorers=[exact_match],
    )

    # Run the eval 5 times, bumping quality each round.
    # This is the fake "learning" — in reality the loop would read the
    # previous score and decide what to change. Here we just turn the knob.
    for iteration in range(5):
        print(f"\n=== Iteration {iteration} | quality={model.quality:.2f} ===")
        results = await evaluation.evaluate(model)
        print(results)

        # "Improve" for next round
        model.quality = min(1.0, model.quality + 0.15)

    print("\nDone. Open the Evals tab in Weave to see the score climb.")


if __name__ == "__main__":
    asyncio.run(main())