import os


def get_sample_from_url(commit_url, label=None):

    sample_path = os.path.join(os.path.dirname(__file__),
                               "15f39fc49d9f25bca872badb48e708a8bb815ea7")
    method_files = [os.path.join(root, f)
                    for root, dirs, files in os.walk(sample_path)
                    for f in files if f.endswith(".java")]

    data = {}
    for filename in method_files:
        version, method = filename.split("/")[-3:-1]
        with open(filename) as f:
            code = f.readlines()
        if version not in data:
            data[version] = {}
        data[version][method] = code

    if label is not None:
        data["label"] = label

    return data
