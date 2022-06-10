import random
import json

def sample_data(fin, sample, fout):
    with open(fin, "r") as f:
        data = f.readlines()
        out_sample = random.sample(data, 100)
        
    with open(fout, "w") as f:
        for i in out_sample:
            f.write(i)


def add_index(fin, fout):
    all_data = []
    with open(fin, "r") as f:
        data = f.readlines()
        
        for i, d in enumerate(data):
            item = json.loads(d)
            item["index"] = i
            new_item = json.dumps(item)
            all_data.append(new_item)
            
    out_sample = random.sample(all_data, 100)
    with open(fout, "w") as f:
        for i in out_sample:
            f.write(i + "\n")

if __name__ == '__main__':
    
    
    # sample data
    
    fin = "../data/tokenize_data/test.tokenize"    
    
    fout = "sample_data/test_sample"
    add_index(fin, fout)

    # sample_data(fin, 100, fout)
