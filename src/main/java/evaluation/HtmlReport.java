package evaluation;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Writes a self-contained HTML evaluation report: confusion
 * matrix (colour-coded), per-class precision / recall / F1,
 * and a gallery of sample test images grouped by true class
 * with their predicted labels and accuracy percentages.
 * Images are embedded as base64 PNG data URIs - no external
 * files needed.
 */
public final class HtmlReport {

    private HtmlReport() {
    }

    /**
     * Writes a full evaluation report.
     *
     * @param outputFile  destination HTML file
     * @param confusion   matrix, [actual][predicted]
     * @param classNames  one name per class
     * @param testImages  test examples, [N][784]
     * @param testLabels  true class per row, [N]
     * @param predicted   predicted class per row, [N]
     * @param scale       image upscale factor (2 = 56x56)
     * @throws IOException on write or encoding failure
     */
    public static void write(Path outputFile,
                             int[][] confusion,
                             String[] classNames,
                             float[][] testImages,
                             int[] testLabels,
                             int[] predicted,
                             int scale)
            throws IOException {
        int numClasses = confusion.length;
        float accuracy = ClassificationMetrics.accuracy(
            predicted, testLabels);

        StringBuilder html = new StringBuilder();

        html.append("<!doctype html>\n<html lang=\"en\">\n");
        html.append("<head>\n");
        html.append("<meta charset=\"utf-8\">\n");
        html.append("<title>Evaluation report</title>\n");
        html.append("<style>").append(css());
        html.append("</style>\n</head>\n<body>\n");

        // Header
        html.append("<h1>Evaluation report</h1>\n");
        html.append(String.format(
            "<p class=\"summary\">Test accuracy: "
            + "<strong>%.2f%%</strong> "
            + "(%d / %d)</p>\n",
            accuracy * 100,
            correctCount(predicted, testLabels),
            testLabels.length));

        // Confusion matrix
        appendConfusion(html, confusion, classNames);

        // Per-class metrics
        appendMetrics(html, confusion, classNames);

        // Per-class accuracy + sample images
        appendClassGalleries(html, testImages, testLabels,
            predicted, classNames, scale);

        html.append("</body>\n</html>\n");

        try (Writer w = Files.newBufferedWriter(outputFile)) {
            w.write(html.toString());
        }
    }

    /**
     * Convenience overload with default scale = 2.
     */
    public static void write(Path outputFile,
                             int[][] confusion,
                             String[] classNames,
                             float[][] testImages,
                             int[] testLabels,
                             int[] predicted)
            throws IOException {
        write(outputFile, confusion, classNames,
              testImages, testLabels, predicted, 2);
    }

    // -- Confusion matrix -------------------------------------

    private static void appendConfusion(StringBuilder html,
                                        int[][] confusion,
                                        String[] names) {
        int n = confusion.length;
        int max = maxCell(confusion);

        html.append("<h2>Confusion matrix</h2>\n");
        html.append("<table class=\"confusion\">\n");
        html.append("<tr><th></th>");

        for (int c = 0; c < n; c++) {
            html.append(String.format("<th>%d</th>", c));
        }
        html.append("</tr>\n");

        for (int a = 0; a < n; a++) {
            html.append(String.format(
                "<tr><th>%s</th>", names[a]));

            for (int p = 0; p < n; p++) {
                int v = confusion[a][p];
                String style = (a == p)
                    ? String.format(
                        "background:hsla(140,60%%,45%%,%.2f)",
                        0.15 + 0.85 * v / max)
                    : "";
                html.append(String.format(
                    "<td style=\"%s\">%d</td>\n",
                    style, v));
            }

            html.append("</tr>\n");
        }

        html.append("</table>\n");
    }

    // -- Per-class metrics ------------------------------------

    private static void appendMetrics(StringBuilder html,
                                      int[][] confusion,
                                      String[] names) {
        html.append("<h2>Per-class metrics</h2>\n");
        html.append("<table>\n");
        html.append("<tr><th>class</th>");
        html.append("<th>precision</th>");
        html.append("<th>recall</th>");
        html.append("<th>f1</th></tr>\n");

        for (int c = 0; c < confusion.length; c++) {
            float p = ClassificationMetrics
                .precision(confusion, c);
            float r = ClassificationMetrics
                .recall(confusion, c);
            float f = ClassificationMetrics.f1(confusion, c);

            html.append(String.format(
                "<tr><td>%s</td>"
                + "<td>%.3f</td>"
                + "<td>%.3f</td>"
                + "<td>%.3f</td></tr>\n",
                names[c], p, r, f));
        }

        html.append("</table>\n");
    }

    // -- Per-class image galleries ----------------------------

    private static void appendClassGalleries(
            StringBuilder html,
            float[][] testImages,
            int[] testLabels,
            int[] predicted,
            String[] names,
            int scale) throws IOException {

        html.append("<h2>Sample predictions by class</h2>\n");
        html.append(
            "<p>Each section shows test images whose "
            + "true label is that class. Green border = "
            + "correct prediction, red = wrong. The accuracy "
            + "shown under each image is the per-class "
            + "accuracy.</p>\n");

        int numClasses = names.length;

        for (int cls = 0; cls < numClasses; cls++) {
            // Per-class accuracy
            int total = 0;
            int correct = 0;

            for (int i = 0; i < testLabels.length; i++) {
                if (testLabels[i] == cls) {
                    total++;
                    if (predicted[i] == cls) correct++;
                }
            }

            float classAcc = total > 0
                ? (float) correct / total : 0.0f;

            html.append(String.format(
                "<div class=\"class-section\">\n"
                + "<h3>%s - accuracy %.2f%% "
                + "(%d / %d)</h3>\n",
                names[cls], classAcc * 100, correct, total));

            // Collect up to 10 sample indices for this class
            List<Integer> samples = new ArrayList<>();

            for (int i = 0; i < testLabels.length
                    && samples.size() < 10; i++) {
                if (testLabels[i] == cls) {
                    samples.add(i);
                }
            }

            html.append("<div class=\"gallery\">\n");

            for (int idx : samples) {
                boolean ok =
                    predicted[idx] == testLabels[idx];
                String uri = ImageEncoder.toDataUri(
                    testImages[idx], scale);
                String borderClass = ok ? "ok" : "bad";

                html.append(String.format(
                    "<figure class=\"%s\">\n"
                    + "<img src=\"%s\" alt=\"%s\">\n"
                    + "<figcaption>"
                    + "<span class=\"pred\">%s</span>\n"
                    + "<span class=\"acc\">%.1f%%</span>"
                    + "</figcaption>\n</figure>\n",
                    borderClass, uri, names[cls],
                    names[predicted[idx]],
                    classAcc * 100));
            }

            html.append("</div>\n</div>\n");
        }
    }

    // -- Helpers ----------------------------------------------

    private static int correctCount(int[] predicted,
                                    int[] labels) {
        int c = 0;
        for (int i = 0; i < labels.length; i++) {
            if (predicted[i] == labels[i]) c++;
        }
        return c;
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

    // -- CSS --------------------------------------------------

    private static String css() {
        return "body{font-family:system-ui,sans-serif;"
            + "margin:2rem;background:#fafafa;color:#222}"
            + "h1{border-bottom:2px solid #333;"
            + "padding-bottom:.3rem}"
            + ".summary{font-size:1.2rem;background:#e8f5e9;"
            + "display:inline-block;padding:.4rem 1rem;"
            + "border-radius:6px}"
            + "table{border-collapse:collapse;margin:1rem 0}"
            + "td,th{border:1px solid #ccc;padding:4px 10px;"
            + "text-align:right}"
            + "th{background:#f0f0f0}"
            + ".confusion td{text-align:right;min-width:3em}"
            + ".class-section{margin:2rem 0;padding:1rem;"
            + "background:#fff;border-radius:8px;"
            + "box-shadow:0 1px 3px rgba(0,0,0,.1)}"
            + ".class-section h3{margin-top:0;color:#333}"
            + ".gallery{display:flex;flex-wrap:wrap;gap:10px}"
            + ".gallery figure{margin:0;padding:6px;"
            + "border-radius:6px;text-align:center;"
            + "background:#f9f9f9}"
            + ".gallery .ok{border:2px solid #4caf50}"
            + ".gallery .bad{border:2px solid #f44336}"
            + ".gallery img{display:block;"
            + "image-rendering:pixelated}"
            + ".gallery figcaption{margin-top:4px;"
            + "font-size:.75rem}"
            + ".gallery .pred{display:block;font-weight:600}"
            + ".gallery .acc{display:block;color:#666}";
    }
}
