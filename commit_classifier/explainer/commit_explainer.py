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
          "<link rel=\"stylesheet\" href=\"highlight.css\" type=\"text/css\">",
          "</head>"]
hor_line = "<div style=\"border-top: 1px solid; margin-top: 5px; " \
           "padding-top: 5px; display: inline-block\">"
pre_color = "<mark style=\"background-color: {}; " \
            "\"white-space: pre\"" \
            "opacity:1.0; line-height:1.75\">"
post_color = "</mark>"
style = ["<style type=\"text/css\">",
         "pre { line-height: 125%; }",
         "td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }",
         "span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }",
         "td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }",
         "span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }",
         "body .hll { background-color: #ffffcc }",
         "body { background: #f8f8f8; }",
         "body .c { color: #3D7B7B; font-style: italic } /* Comment */",
         "body .err { border: 1px solid #FF0000 } /* Error */",
         "body .k { color: #008000; font-weight: bold } /* Keyword */",
         "body .o { color: #666666 } /* Operator */",
         "body .ch { color: #3D7B7B; font-style: italic } /* Comment.Hashbang */",
         "body .cm { color: #3D7B7B; font-style: italic } /* Comment.Multiline */",
         "body .cp { color: #9C6500 } /* Comment.Preproc */",
         "body .cpf { color: #3D7B7B; font-style: italic } /* Comment.PreprocFile */",
         "body .c1 { color: #3D7B7B; font-style: italic } /* Comment.Single */",
         "body .cs { color: #3D7B7B; font-style: italic } /* Comment.Special */",
         "body .gd { color: #A00000 } /* Generic.Deleted */",
         "body .ge { font-style: italic } /* Generic.Emph */",
         "body .gr { color: #E40000 } /* Generic.Error */",
         "body .gh { color: #000080; font-weight: bold } /* Generic.Heading */",
         "body .gi { color: #008400 } /* Generic.Inserted */",
         "body .go { color: #717171 } /* Generic.Output */",
         "body .gp { color: #000080; font-weight: bold } /* Generic.Prompt */",
         "body .gs { font-weight: bold } /* Generic.Strong */",
         "body .gu { color: #800080; font-weight: bold } /* Generic.Subheading */",
         "body .gt { color: #0044DD } /* Generic.Traceback */",
         "body .kc { color: #008000; font-weight: bold } /* Keyword.Constant */",
         "body .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */",
         "body .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */",
         "body .kp { color: #008000 } /* Keyword.Pseudo */",
         "body .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */",
         "body .kt { color: #B00040 } /* Keyword.Type */",
         "body .m { color: #666666 } /* Literal.Number */",
         "body .s { color: #BA2121 } /* Literal.String */",
         "body .na { color: #687822 } /* Name.Attribute */",
         "body .nb { color: #008000 } /* Name.Builtin */",
         "body .nc { color: #0000FF; font-weight: bold } /* Name.Class */",
         "body .no { color: #880000 } /* Name.Constant */",
         "body .nd { color: #AA22FF } /* Name.Decorator */",
         "body .ni { color: #717171; font-weight: bold } /* Name.Entity */",
         "body .ne { color: #CB3F38; font-weight: bold } /* Name.Exception */",
         "body .nf { color: #0000FF } /* Name.Function */",
         "body .nl { color: #767600 } /* Name.Label */",
         "body .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */",
         "body .nt { color: #008000; font-weight: bold } /* Name.Tag */",
         "body .nv { color: #19177C } /* Name.Variable */",
         "body .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */",
         "body .w { color: #bbbbbb } /* Text.Whitespace */",
         "body .mb { color: #666666 } /* Literal.Number.Bin */",
         "body .mf { color: #666666 } /* Literal.Number.Float */",
         "body .mh { color: #666666 } /* Literal.Number.Hex */",
         "body .mi { color: #666666 } /* Literal.Number.Integer */",
         "body .mo { color: #666666 } /* Literal.Number.Oct */",
         "body .sa { color: #BA2121 } /* Literal.String.Affix */",
         "body .sb { color: #BA2121 } /* Literal.String.Backtick */",
         "body .sc { color: #BA2121 } /* Literal.String.Char */",
         "body .dl { color: #BA2121 } /* Literal.String.Delimiter */",
         "body .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */",
         "body .s2 { color: #BA2121 } /* Literal.String.Double */",
         "body .se { color: #AA5D1F; font-weight: bold } /* Literal.String.Escape */",
         "body .sh { color: #BA2121 } /* Literal.String.Heredoc */",
         "body .si { color: #A45A77; font-weight: bold } /* Literal.String.Interpol */",
         "body .sx { color: #008000 } /* Literal.String.Other */",
         "body .sr { color: #A45A77 } /* Literal.String.Regex */",
         "body .s1 { color: #BA2121 } /* Literal.String.Single */",
         "body .ss { color: #19177C } /* Literal.String.Symbol */",
         "body .bp { color: #008000 } /* Name.Builtin.Pseudo */",
         "body .fm { color: #0000FF } /* Name.Function.Magic */",
         "body .vc { color: #19177C } /* Name.Variable.Class */",
         "body .vg { color: #19177C } /* Name.Variable.Global */",
         "body .vi { color: #19177C } /* Name.Variable.Instance */",
         "body .vm { color: #19177C } /* Name.Variable.Magic */",
         "body .il { color: #666666 } /* Literal.Number.Integer.Long */",
         "</style>"]


class CommitExplainer:

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
        dom += style
        dom.append(hor_line)
        dom += ["<p><b>True Label:</b> {}</p>".format(true_class),
                "<p><b>Predicted Label:</b> {}</p>".format(attr_class),
                "<p><b>Score:</b> {:.6f}</p>".format(
                    float(self.explainer.pred_probs))]
        if commit_url:
            dom.append("<p><b>Commit url:</b><a href=\"{}\"> {}</a></p>"
                       .format(commit_url, commit_url))
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

        return {i: [(t[0], normalize(t[1])) for t in l]
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
                    for tok, imp in explained_lines[i]:
                        tok = tok.strip("##")
                        if tok == word[:len(tok)].lower():
                            color = self._get_color(imp)
                            formatted_line = formatted_line[:s+offset] + \
                                pre_color.format(color) + word[:len(tok)] + \
                                post_color + formatted_line[e+offset-l_word:]
                            word = word[len(tok):]
                            offset += len(pre_color.format(color)) + len(tok) \
                                + len(post_color)
                    offset -= l_word
            tags.append(formatted_line)
        return "\n".join(tags)
