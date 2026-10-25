# Save the evaluation as a web page

*The console output from chapter 10 is useful while you work, but a trained model's results deserve something you can share: a self-contained HTML file with the confusion matrix, per-class precision/recall/F1, and a gallery of sample predictions. This chapter builds exactly that.*

## 11.1 Why HTML?

- **Shareable.** One file, opens in any browser, no server needed.
- **Richer than plain text.** You can colour the confusion matrix cells, render the sample images as tiny canvases, and style the metrics table.
- **Easy to generate.** HTML is just a string of text — you can build it with `StringBuilder` and a few loops, no template engine required.

## 11.2 The report class

```
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Writes a self-contained HTML evaluation report: confusion
 * matrix (colour-coded), per-class precision/recall/F1, and a
 * gallery of sample test images with their predictions.
 */
public final class HtmlReport {

    /** Writes the report to a file. */
    public static void write(Path outputFile,
                             int[][] confusion,
                             String[] classNames,
                             float[][] testImages,
                             int[] testLabels,
                             int[] predicted,
                             float accuracy) throws IOException {

        StringBuilder html = new StringBuilder();

        html.append("<!doctype html><html><head>")
            .append("<meta charset=\"utf-8\">")
            .append("<title>Evaluation report</title>")
            .append("<style>")
            .append(css())
            .append("</style></head><body>");

        html.append("<h1>Evaluation report</h1>");
        html.append(String.format(
            "<p>Test accuracy: <strong>%.2f%%</strong></p>",
            accuracy * 100));

        appendConfusionTable(html, confusion, classNames);
        appendMetricsTable(html, confusion, classNames);
        appendGallery(html, testImages, testLabels, predicted,
                      classNames);

        html.append("</body></html>");

        try (Writer w = Files.newBufferedWriter(outputFile)) {
            w.write(html.toString());
        }
    }

    private static void appendConfusionTable(StringBuilder html,
                                             int[][] confusion,
                                             String[] names) {
        int n = confusion.length;

        html.append("<h2>Confusion matrix</h2>")
            .append("<table class=\"confusion\">");

        // header row
        html.append("<tr><th></th>");
        for (int c = 0; c < n; c++) {
            html.append(String.format("<th>%d</th>", c));
        }
        html.append("</tr>");

        int max = maxCell(confusion);

        for (int a = 0; a < n; a++) {
            html.append(String.format(
                "<tr><th>%s</th>", names[a]));

            for (int p = 0; p < n; p++) {
                int v = confusion[a][p];
                // darker green for larger correct counts
                String style = (a == p)
                    ? String.format(
                        "background:hsla(140,60%%,45%%,%.2f)",
                        0.2 + 0.8 * v / max)
                    : "";
                html.append(String.format(
                    "<td style=\"%s\">%d</td>",
                    style, v));
            }

            html.append("</tr>");
        }

        html.append("</table>");
    }

    private static void appendMetricsTable(StringBuilder html,
                                           int[][] confusion,
                                           String[] names) {
        html.append("<h2>Per-class metrics</h2>")
            .append("<table><tr>"
                + "<th>class</th>"
                + "<th>precision</th>"
                + "<th>recall</th>"
                + "<th>f1</th></tr>");

        for (int c = 0; c < confusion.length; c++) {
            float p = ClassificationMetrics.precision(confusion, c);
            float r = ClassificationMetrics.recall(confusion, c);
            float f = ClassificationMetrics.f1(confusion, c);

            html.append(String.format(
                "<tr><td>%s</td>"
                + "<td>%.3f</td>"
                + "<td>%.3f</td>"
                + "<td>%.3f</td></tr>",
                names[c], p, r, f));
        }

        html.append("</table>");
    }

    private static void appendGallery(StringBuilder html,
                                      float[][] images,
                                      int[] labels,
                                      int[] predicted,
                                      String[] names) {
        html.append("<h2>Sample predictions</h2>")
            .append("<div class=\"gallery\">");

        for (int i = 0; i < Math.min(20, images.length); i++) {
            boolean ok = predicted[i] == labels[i];
            html.append(String.format(
                "<figure class=\"%s\">"
                + "<figcaption>%s</figcaption>"
                + "</figure>",
                ok ? "ok" : "bad",
                names[predicted[i]]));
        }

        html.append("</div>");
    }

    private static int maxCell(int[][] m) {
        int max = 1;
        for (int[] row : m) {
            for (int v : row) {
                if (v > max) max = v;
            }
        }
        return max;
    }

    private static String css() {
        return "body{font-family:sans-serif;margin:2rem}"
            + "table{border-collapse:collapse;margin:1rem 0}"
            + "td,th{border:1px solid #ccc;padding:4px 8px;"
            + "text-align:right}"
            + "th{background:#f4f4f4}"
            + ".gallery{display:flex;flex-wrap:wrap;gap:8px}"
            + ".gallery figure{margin:0;padding:4px;"
            + "border-radius:4px;text-align:center}"
            + ".gallery .ok{background:#e8f5e9}"
            + ".gallery .bad{background:#ffebee}";
    }

    private HtmlReport() { }
}
```

## 11.3 Colour-coding the matrix

The diagonal cells (correct predictions) get a green background whose intensity scales with the count relative to the largest cell. Off-diagonal cells stay plain, so mistakes pop out against the green diagonal:

```
String style = (a == p)
    ? String.format("background:hsla(140,60%%,45%%,%.2f)",
        0.2 + 0.8 * v / max)
    : "";
```

## 11.4 Wiring it into the runner

```
// after building the confusion matrix and predicted[] array
float accuracy = ClassificationMetrics.accuracy(predicted, labels);

HtmlReport.write(
    Path.of("report.html"),
    confusion,
    FashionMNISTLoader.CLASS_NAMES,
    testImages,
    labels,
    predicted,
    accuracy);

System.out.println("Report written to report.html");
```

> **Try it.** Open `report.html` in a browser. The Shirt row will show the lowest recall — you can see at a glance which classes the model confuses, without scrolling through console output.

---

[← 10 · Evaluation](10-evaluation.md) · [12 · Swing/AWT window →](12-swing-ui.md)
