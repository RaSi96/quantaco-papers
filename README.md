# Quantaco Papers
Welcome! This is a repository of the papers/articles I had published on [Quantaco's Medium tech blog](https://medium.com/@quantaco_rahuls) whilst a (the only) Data Scientist there (2023-2025).

The `src` folder contains subfolders with (almost) all of the papers/articles in Markdown/LaTeX. Wherever appropriate, original resolution images have also been included. Each article _should_ correctly compile into a PDF with this kind of `pandoc` command:

```bash
pandoc FILE.md \\
-f markdown+lists_without_preceding_blankline \\
--pdf-engine=xelatex \\
--wrap=none \\
-o FILE.pdf
```

With perhaps the exception of the lonesome `.odt` file, which I wrote way before I found out about `pandoc` (and hence might be riddled with typesetting errors if one were to try using `pandoc` on it). That one can probably just be exported as a PDF directly from LibreOffice Writer. Datasets and R&D work is, of course, proprietary. The latest compiled PDFs _should_ also be made available via the Releases section of this repo.

This work is licensed by the CC-BY-SA 4.0 license.
