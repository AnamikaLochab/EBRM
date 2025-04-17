# Taken and modified from Coste's(tlc4418) llm_optimization repository
from torch.utils.data import Dataset


def _parse_row_input(row: dict):
    instruction = row["instruction"]
    input = row["input"]
    # print(row)
    # instruction = row['prompt']
    if len(row["input"]) > 0:
        input_ = "{}\n{}".format(instruction, input)
    else:
        input_ = instruction
    # print(instruction)
    # input_ = instruction
    prompt = f"<|prompter|>{input_}<|endoftext|>"
    return prompt


class AlpacaFarmHumanPref(Dataset):
    name = "alpaca_farm_pref"

    def __init__(self, dataset) -> None:
        super().__init__()
        self.data = []

        for row in dataset:
            prompt = _parse_row_input(row)
            outputs = [f"<|assistant|>{o}<|endoftext|>" for o in [row["output_1"], row["output_2"]]]
            preference = row["preference"] - 1
            ordered_outputs = [outputs[preference], outputs[not preference]]
            self.data.append(([prompt], ordered_outputs))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, list[str]]:
        return self.data[index]


class CustomHFPref(Dataset):
    name = "custom_hf_pref"

    def __init__(self, dataset, stop, train=True) -> None:
        super().__init__()
        self.data = []
        self.stop = stop

        for i, row in enumerate(dataset):
            if i == stop:
                break
            prompt = _parse_row_input(row)
            outputs = [f"<|assistant|>{o}<|endoftext|>" for o in row["answers"]]
            preference = row["preference"]  # if (not train or i % 2 == 0) else random.randint(0, 1)
            ordered_outputs = [outputs[preference], outputs[not preference]]
            self.data.append(([prompt], ordered_outputs))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, list[str]]:
        return self.data[index]

class CustomUFPref(Dataset):
    name = "custom_uf_pref"

    def __init__(self, dataset, stop, train=True, score = False) -> None:
        super().__init__()
        self.data = []
        self.stop = stop
        # print(dataset[0])
        if score:
            for i, row in enumerate(dataset):

                if i == stop:
                    break
                prompt = _parse_row_input(row)
                outputs = [f"<|assistant|>{row['chosen'][1]['content']}<|endoftext|>" , f"<|assistant|>{row['rejected'][1]['content']}<|endoftext|>"  ]
                scores = [row['score_chosen'], row['score_rejected']]
                preference = 0  # if (not train or i % 2 == 0) else random.randint(0, 1)
                ordered_outputs = [outputs[preference], outputs[not preference]]
                scored_outputs = [scores[preference], scores[not preference]]
                self.data.append(([prompt], ordered_outputs, scored_outputs))
        else:

            for i, row in enumerate(dataset):
                if i == stop:
                    break
                prompt = _parse_row_input(row)
                outputs = [f"<|assistant|>{row['chosen'][1]['content']}<|endoftext|>" , f"<|assistant|>{row['rejected'][1]['content']}<|endoftext|>"  ]
                preference = 0  # if (not train or i % 2 == 0) else random.randint(0, 1)
                ordered_outputs = [outputs[preference], outputs[not preference]]
                self.data.append(([prompt], ordered_outputs))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, list[str]]:
        return self.data[index]
