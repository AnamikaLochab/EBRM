# Taken from Coste's(tlc4418) llm_optimization repository 
import typer
from src.ppo.custom_helpers import gold_score

if __name__ == "__main__":
    typer.run(gold_score)
