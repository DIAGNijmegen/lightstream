window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
  },
  options: {
    ignoreHtmlClass: "[\\s\\S]*",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => MathJax.typesetPromise());
