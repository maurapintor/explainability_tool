from commit_classifier.explainer.explainer import \
    JavaBERTCommitClassificationExplainer
import difflib
import pygments
from pygments import lexers, formatters
import re
import numpy as np
import html


header = ["<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01//EN\"",
          "\"http://www.w3.org/TR/html4/strict.dtd\">",
          "<head>",
          "<title>Explainability Tool</title>",
          "<link rel=\"stylesheet\" href=\"static/styles.css\" type=\"text/css\">",
          "<link id=\"favicon\" rel=\"icon\" type=\"image/x-icon\" href=\"assets/images/favicon.ico\">",
          "<link rel=\"apple-touch-icon\" type=\"image/x-icon\" href=\"assets/images/favicon.ico\">",
          "<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js\" "
          "integrity=\"sha384-geWF76RCwLtnZ8qwWowPQNguL3RmwHVBC9FhGdlKrxdiJJigb/j/68SIy3Te4Bkz\" "
          "crossorigin=\"anonymous\"></script>",
          "</head>",
          "<div>"
          "<div class=\"row\">",
          "<div class=\"col-sm-1\">",
          "<img height=\"80px\" src=\"assets/images/assuremoss_logo.png\" alt=\"AssureMOSS logo\">",
          "</div>",
          "<div class=\"col\">",
          "<h1>Explainability tool</h1>",
          "</div>",
          "</div>"
          "</div>"]
hor_line = "<div style=\"border-top: 1px solid; margin-top: 5px; " \
           "padding-top: 5px; display: inline-block\">"
button = "<button type=\"button\" id=\"btn\" onclick=\"function show() {" \
         "document.getElementById('explanation').style.display = ''; " \
         "document.getElementById('btn').style.display = 'none';}; show();\">" \
         "Explain</button>"
pre_color = "<mark title=\"{:.6f}\" style=\"background-color: {}; " \
            "\"white-space: pre\"" \
            "opacity:1.0; line-height:1.75\">"
post_color = "</mark>"


class CommitExplainer:

    """some parts are adapted from https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py"""

    def __init__(self, model, tokenizer, custom_labels=None):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = JavaBERTCommitClassificationExplainer(
            model, tokenizer, custom_labels=custom_labels)
        self._d = difflib.Differ()
        self._prev_token = ""
        self._lexer = lexers.get_lexer_by_name("Java")
        self._formatter = formatters.HtmlFormatter(
            cssstyles="font-size: 16px;")

    def explain(self, commit):

        pre, post = commit["pre"], commit["post"]

        diff_lines = []
        pre_diffs = []
        post_diffs = []
        for method in pre.keys():
            pre_method, post_method = pre[method], post[method]
            method_pre_diffs = []
            method_post_diffs = []
            diffs = self._d.compare(pre_method, post_method)
            for line in diffs:
                diff_lines.append(line)
                if line.startswith("- "):
                    method_pre_diffs.append(line[2:])
                elif line.startswith("+ "):
                    method_post_diffs.append(line[2:])
            pre_diffs.append(method_pre_diffs)
            post_diffs.append(method_post_diffs)

        concat_diffs = (" ".join([" ".join(line.split())
                                  for m in pre_diffs for line in m]),
                        " ".join([" ".join(line.split())
                                  for m in post_diffs for line in m]))
        word_attributions = self.explainer(*concat_diffs)

        tokens, values = map(list, zip(*word_attributions))
        idx_1 = tokens.index(self.tokenizer.sep_token)
        idx_2 = len(tokens) - 1 - tokens[::-1].index(self.tokenizer.sep_token)
        pre_tokens, pre_values = tokens[1:idx_1], values[1:idx_1]
        pre_word_attributions = word_attributions[1:idx_1]
        post_tokens, post_values = tokens[idx_1 + 1:idx_2], values[
                                                            idx_1 + 1:idx_2]
        post_word_attributions = word_attributions[idx_1 + 1:idx_2]

        explained_lines = {}
        s_idx_pre, s_idx_post = 0, 0
        for i, diff_line in enumerate(diff_lines):
            if diff_line.startswith("- "):
                diff_line = " ".join(diff_line[2:].split())
                tokenized = self.tokenizer.tokenize(diff_line)
                e_idx_pre = s_idx_pre + len(tokenized)
                if not pre_tokens[s_idx_pre:e_idx_pre]:
                    continue
                if len(pre_tokens[s_idx_pre:e_idx_pre]) < len(tokenized):
                    tokenized = tokenized[:len(pre_tokens[
                                               s_idx_pre:e_idx_pre])]
                if all(t1 == t2 for t1, t2 in zip(
                        pre_tokens[s_idx_pre:e_idx_pre], tokenized)):
                    explained_lines[i] = pre_word_attributions[s_idx_pre:
                                                               e_idx_pre]
                    s_idx_pre = e_idx_pre
            elif diff_line.startswith("+ "):
                diff_line = " ".join(diff_line[2:].split())
                tokenized = self.tokenizer.tokenize(diff_line)
                e_idx_post = s_idx_post + len(tokenized)
                if not post_tokens[s_idx_post:e_idx_post]:
                    continue
                if len(post_tokens[s_idx_post:e_idx_post]) < len(tokenized):
                    tokenized = tokenized[:len(post_tokens[
                                               s_idx_post:e_idx_post])]
                if all(t1 == t2 for t1, t2 in zip(
                        post_tokens[s_idx_post:e_idx_post], tokenized)):
                    explained_lines[i] = post_word_attributions[s_idx_post:
                                                                e_idx_post]
                    s_idx_post = e_idx_post

        return diff_lines, explained_lines

    def explain_visualize(self, commit, html_filepath=None, true_class=None,
                          legend=True, commit_url=None):
        return self.visualize(*self.explain(commit),
                              html_filepath, true_class, legend, commit_url)

    def visualize(self, diff_lines, explained_lines, html_filepath=None,
                  true_class=None, legend=True, commit_url=None):

        attr_class = self.explainer.id2label[self.explainer.selected_index]
        true_class = "not available" if true_class is None else true_class

        try:
            explained_lines = self._normalize_attributions(explained_lines)
        except ValueError:
            raise Exception(f"Cannot visualize explanation (commit url: "
                            f"{commit_url})")

        dom = []
        dom += header
        dom += ["<p><b>True Label:</b> {}</p>".format(true_class),
                "<p><b>Predicted Label:</b> {}</p>".format(attr_class),
                "<p><b>Score:</b> {:.6f}</p>".format(
                    float(self.explainer.pred_probs))]
        if commit_url:
            dom.append("<p><b>Commit url:</b><a href=\"{}\"> {}</a></p>"
                       .format(commit_url, commit_url))
        dom.append(button)
        dom.append("<div id=\"explanation\" style=\"display: none;\">")
        dom.append(hor_line)
        if legend:
            dom.append("<b>Legend: </b>")
            for value, label in zip(
                    [-1, 0, 1], [self.explainer.id2label[0], "Neutral",
                                 self.explainer.id2label[1]]):
                dom.append(
                    "<span style=\"display: inline-block; width: 10px;"
                    "height: 10px; border: 1px solid; background-color: "
                    "{value}\"></span> {label}  ".format(
                        value=self._get_color(value), label=label))
            dom.append("<hr><br>")
        dom.append(self._format_code_lines(diff_lines, explained_lines))
        dom.append("</div>")
        html_txt = "\n".join(dom)

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html_txt)
        return html_txt

    def _get_color(self, attr):
        # clip values to prevent CSS errors (Values should be from [-1,1])
        attr = max(-1, min(1, attr))
        if attr > 0:
            hue = 120
            sat = 75
            lig = 100 - int(50 * attr)
        else:
            hue = 0
            sat = 75
            lig = 100 - int(-40 * attr)
        return "hsl({}, {}%, {}%)".format(hue, sat, lig)

    def _normalize_attributions(self, explained_lines):

        attrs = np.array([t[1] for l in explained_lines.values() for t in l])
        min_pos, max_pos = attrs[attrs > 0].min(), attrs[attrs > 0].max()
        min_neg, max_neg = attrs[attrs < 0].min(), attrs[attrs < 0].max()

        def normalize(value):
            if value > 0:
                r_min = min_pos
                r_max = max_pos
                n_min = 0
                n_max = 1
            elif value < 0:
                r_min = min_neg
                r_max = max_neg
                n_min = -1
                n_max = 0
            else:
                return value
            return (value - r_min) / (r_max - r_min) * (n_max - n_min) + n_min

        return {i: [(t[0], t[1], normalize(t[1])) for t in l]
                for i, l in explained_lines.items()}

    def _format_code_lines(self, diff_lines, explained_lines):
        tags = []
        for i, diff_line in enumerate(diff_lines):
            formatted_line = html.unescape(pygments.highlight(
                    diff_line.replace("\n", ""), self._lexer,
                    self._formatter))
            if i in explained_lines:
                s_idx = [m.end() for m in
                         re.finditer("<span class=\".*?\">",
                                     formatted_line)][1:]
                e_idx = [m.start() for m in
                         re.finditer("</span>", formatted_line)][2:]
                offset = 0
                for s, e in zip(s_idx, e_idx):
                    word = formatted_line[s+offset:e+offset]
                    formatted_line = formatted_line[:s+offset] + \
                        formatted_line[e+offset:]
                    l_word = len(word)
                    word = word.replace(" ", "")
                    for tok, attr, norm_attr in explained_lines[i]:
                        tok = tok.strip("##")
                        if tok == word[:len(tok)].lower():
                            color = self._get_color(norm_attr)
                            formatted_line = formatted_line[:s+offset] + \
                                pre_color.format(attr, color) + \
                                word[:len(tok)] + post_color + \
                                formatted_line[e+offset-l_word:]
                            word = word[len(tok):]
                            offset += len(pre_color.format(attr, color)) + \
                                len(tok) + len(post_color)
                    offset -= l_word
            tags.append(formatted_line)
        return "\n".join(tags)
