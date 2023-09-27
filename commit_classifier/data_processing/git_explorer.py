import os
import re
from pydriller.git import Git, Repo
import pathlib
from urllib.parse import urlparse
import signal


pattern = r"(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')|(/\*.*?\*/|//[^\r\n]*$)"
regex = re.compile(pattern, re.MULTILINE | re.DOTALL)


def _replacer(match):
    if match.group(2) is not None:
        return ""
    elif match.group(1) is not None:
        return match.group(1)


class TimeOut:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def path_from_url(url, base_path):
    url = url.rstrip("/")
    parsed_url = urlparse(url)
    return os.path.join(
        base_path, parsed_url.netloc + parsed_url.path.replace("/", "_"))


class GitExplorer:

    def __init__(self, cache_path=os.path.abspath("/tmp/gitcache")):

        self.cache_path = cache_path
        pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)

    def _get_repo(self, url, timeout=60):
        repo_path = path_from_url(url, self.cache_path)
        if not os.path.isdir(repo_path):
            with TimeOut(seconds=timeout):
                Repo.clone_from(str(url), repo_path)
        return Git(repo_path)

    def get_modified_methods(self, repo_url, commit_id, max_len=1000):

        try:
            repo = self._get_repo(repo_url)
        except Exception as repo_ex:
            raise Exception(f"There was an error with repository {repo_url}: "
                            f"{str(repo_ex)}")

        try:
            data = {"pre": {}, "post": {}}

            try:
                commit = repo.get_commit(commit_id)
            except Exception:
                raise Exception("Commit not found")

            changed_files = [f for f in commit.modified_files if
                             os.path.splitext(f.filename)[1] == ".java"
                             and len(f.changed_methods) != 0]
            if len(changed_files) == 0:
                raise Exception(f"No modified java files or methods in "
                                f"commit {commit_id}")

            # Iterate over changed files
            for file in changed_files:

                modified_method_names = [mtd.long_name for mtd in
                                         file.changed_methods]

                before_methods = {
                    mtd.long_name: mtd for mtd in file.methods_before
                    if mtd.long_name in modified_method_names}
                after_methods = {
                    mtd.long_name: mtd for mtd in file.methods if
                    mtd.long_name in before_methods.keys()}

                for method_name, idx in zip(after_methods,
                                            range(len(after_methods))):
                    m_after = after_methods[method_name]
                    m_before = before_methods[method_name]
                    method_after_code = self._get_method_code(
                        m_after, file.source_code)
                    method_before_code = self._get_method_code(
                        m_before, file.source_code_before)
                    if len(method_before_code.split('\n')) > max_len \
                            or len(method_after_code.split('\n')) > max_len:
                        continue

                    annotations = '\n'.join(file.source_code.split('\n')[
                                            m_after.start_line - 2:
                                            m_after.start_line - 1])
                    if '@Test' in annotations:
                        continue

                    m_filename = f'{file.filename}@method{idx}'
                    if re.sub(r'\s+', '', method_after_code) != \
                            re.sub(r'\s+', '', method_before_code):
                        data["pre"][m_filename] = [
                            l for l in method_before_code.split("\n")]
                        data["post"][m_filename] = [
                            l for l in method_after_code.split("\n")]
            return data

        except Exception as cmt_ex:
            raise Exception(f"There was an error with commit {commit_id}: "
                            f"{str(cmt_ex)}")

    @staticmethod
    def _get_method_code(m, source_code):
        method_code = "\n".join(source_code.split('\n')[m.start_line - 1:
                                                        m.end_line])
        method_code = regex.sub(_replacer, method_code)
        method_code = re.sub(r'(\s*\n)', '\n', method_code)
        m_class = m.long_name[:m.long_name.find('(')].split('::')[:-1]

        for i, cl in enumerate(m_class[::-1]):
            white_space = ''.join(['  '] * (len(m_class) - 1 - i))
            c_header = white_space + 'class ' + cl + ' {\n'
            method_code = c_header + method_code + '\n' + white_space + '}'
        return method_code
