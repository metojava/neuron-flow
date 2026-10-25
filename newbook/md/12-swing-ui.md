# Show images and predictions in a window

*Numbers in a console tell you the accuracy; seeing the actual 28×28 images next to the model's predictions builds intuition. This chapter renders a grid of test images in a Swing window, colour-coded by whether the prediction was correct.*

## 12.1 Scaling a 28×28 float image to a BufferedImage

Each test image is a `float[784]` with values in [0, 1]. To display it we map each pixel to a grayscale `int` in a `BufferedImage`, then scale it up with nearest-neighbour so it is visible:

```
import java.awt.image.BufferedImage;

/**
 * Converts one [0,1] float image to an upscaled grayscale
 * BufferedImage for display.
 *
 * @param pixels 784 floats in [0, 1]
 * @param scale  integer upscale factor (e.g. 4 → 112×112)
 * @return a TYPE_BYTE_GRAY BufferedImage
 */
public static BufferedImage toImage(float[] pixels, int scale) {
    int w = 28 * scale;
    BufferedImage img = new BufferedImage(
        w, w, BufferedImage.TYPE_BYTE_GRAY);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float v = pixels[y * 28 + x];
            int gray = Math.round(v * 255.0f);

            for (int dy = 0; dy < scale; dy++) {
                for (int dx = 0; dx < scale; dx++) {
                    img.setRGB(x * scale + dx,
                               y * scale + dy,
                               (gray << 16) | (gray << 8) | gray);
                }
            }
        }
    }

    return img;
}
```

## 12.2 A grid panel of labelled samples

```
import javax.swing.JComponent;
import javax.swing.JPanel;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;

/**
 * A JPanel showing a fixed grid of test images, each with a
 * border coloured green (correct) or red (wrong prediction).
 */
public class SamplePanel extends JPanel {

    private final BufferedImage[] images;
    private final String[] captions;
    private final boolean[] correct;

    public SamplePanel(BufferedImage[] images,
                       String[] captions,
                       boolean[] correct) {
        this.images = images;
        this.captions = captions;
        this.correct = correct;
        setLayout(new GridLayout(2, 10, 4, 4));
        setBackground(Color.DARK_GRAY);

        for (int i = 0; i < images.length; i++) {
            add(new Cell(images[i], captions[i], correct[i]));
        }
    }

    /** One image cell with a coloured border and caption. */
    private static class Cell extends JComponent {
        private final BufferedImage img;
        private final String caption;
        private final boolean ok;

        Cell(BufferedImage img, String caption, boolean ok) {
            this.img = img;
            this.caption = caption;
            this.ok = ok;
            setPreferredSize(
                new java.awt.Dimension(img.getWidth(),
                    img.getHeight() + 18));
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.drawImage(img, 0, 0, null);
            g2.setColor(ok ? Color.GREEN : Color.RED);
            g2.drawRect(0, 0, img.getWidth() - 1,
                        img.getHeight() - 1);
            g2.setColor(Color.WHITE);
            g2.drawString(caption, 2, img.getHeight() + 14);
        }
    }
}
```

## 12.3 Putting it in a window

```
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

/**
 * Builds the sample panel on the Swing event-dispatch thread
 * and shows it in a JFrame.
 */
public static void showWindow(float[][] testImages,
                              int[] labels,
                              int[] predicted,
                              String[] classNames) {
    int n = Math.min(20, testImages.length);

    BufferedImage[] imgs = new BufferedImage[n];
    String[] captions = new String[n];
    boolean[] ok = new boolean[n];

    for (int i = 0; i < n; i++) {
        imgs[i]     = toImage(testImages[i], 4);
        captions[i] = classNames[predicted[i]];
        ok[i]       = predicted[i] == labels[i];
    }

    SwingUtilities.invokeLater(() -> {
        JFrame frame = new JFrame("Fashion-MNIST predictions");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new SamplePanel(imgs, captions, ok));
        frame.pack();
        frame.setVisible(true);
    });
}
```

> **Why SwingUtilities.invokeLater?** Swing components must be created and shown on the event-dispatch thread, not the thread that trained the model. Wrapping the frame setup in `invokeLater` schedules it onto that thread safely.

## 12.4 Wiring it into the runner

```
// after computing predicted[] on the test set
showWindow(testImages, testLabels, predicted,
           FashionMNISTLoader.CLASS_NAMES);
```

The green-bordered cells are correct predictions; red ones are mistakes. In a typical run you will see most reds cluster around Shirt / Pullover / Coat — exactly what the confusion matrix from chapter 10 showed numerically.

---

[← 11 · HTML report](11-html-report.md) · [13 · Run it →](13-run.md)
