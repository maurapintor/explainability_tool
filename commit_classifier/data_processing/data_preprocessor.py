from .git_explorer import GitExplorer


def get_sample_from_url(commit_url, label=None):

    repo_url, commit_id = commit_url.split("/commit/")

    data = GitExplorer().get_modified_methods(repo_url, commit_id)

    if label is not None:
        data["label"] = label

    return data
