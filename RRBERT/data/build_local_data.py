import json
from tqdm import tqdm


def build(source, destination, rows):
    datasets = []
    count = 0
    with open(source, 'r', encoding='utf-8') as f:
        for lidx, line in enumerate(tqdm(f)):
            if count < rows:
                sample = json.loads(line.strip())
                datasets.append(sample)
                count += 1
            else:
                break
    f.close()
    with open(destination, 'w', encoding='utf-8') as f:
        json.dump(datasets, f, ensure_ascii=False)
    f.close()

if __name__ == "__main__":
    # build("./search.train.json", "./search.train.local.json", 20000)
    # build("./zhidao.train.json", "./zhidao.train.local.json", 20000)
    build("./search.dev.json", "./search.dev.local.json", 4000)
    build("./zhidao.dev.json", "./zhidao.dev.local.json", 4000)
    # build("./search.test.json", "./search.test.local.json", 500)
    # build("./zhidao.test.json", "./zhidao.test.local.json", 500)