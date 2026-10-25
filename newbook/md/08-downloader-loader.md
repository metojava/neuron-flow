# Fetching and parsing Fashion MNIST

*Two classes handle data: `FashionMNISTDownloader` downloads four `.gz` files from GitHub and decompresses them; `FashionMNISTLoader` parses the IDX bytes into ready-to-train arrays.*

## 8.1 FashionMNISTDownloader

The source URL points at the official repository:

```
https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/
```

### The files it fetches

```
train-images-idx3-ubyte.gz   # 60,000 training images
train-labels-idx1-ubyte.gz   # 60,000 training labels
t10k-images-idx3-ubyte.gz    # 10,000 test images
t10k-labels-idx1-ubyte.gz    # 10,000 test labels
```

### Public API

| Method | Parameters | Returns | Behavior |
| --- | --- | --- | --- |
| `download()` | none | `Path` | Downloads into `data/fashion-mnist`. |
| `download(Path dir)` | target directory | `Path` | Creates the directory, downloads missing files only, skips files that already exist. |
| `main(String[] args)` | optional directory argument | `void` | CLI entry: `java FashionMNISTDownloader [dir]`. |

### The download loop

```
/**
 * Downloads and decompresses the four Fashion-MNIST IDX files.
 * Files already present (size > 0) are skipped, so the call is
 * cheap and safe to repeat.
 *
 * @param directory where the raw IDX files should live
 * @return the same directory, for chaining
 * @throws IOException            on network or filesystem errors
 * @throws InterruptedException   if the thread is interrupted
 */
public static Path download(Path directory)
        throws IOException, InterruptedException {

    Files.createDirectories(directory);

    for (String file : FILES) {
        Path rawFile  = directory.resolve(file);
        Path gzipFile = directory.resolve(file + ".gz");

        if (Files.exists(rawFile) && Files.size(rawFile) > 0) {
            System.out.println("Already available: " + rawFile);
            continue;
        }

        System.out.println("Downloading " + file + ".gz");

        try {
            downloadFile(BASE_URL + file + ".gz", gzipFile);
            decompress(gzipFile, rawFile);
        } finally {
            Files.deleteIfExists(gzipFile);   // never leave .gz behind
        }

        System.out.println("Saved " + rawFile);
    }

    System.out.println("Fashion-MNIST files are ready in: "
        + directory.toAbsolutePath());

    return directory;
}
```

### downloadFile — HTTP with Java 11+ HttpClient

```
/**
 * Streams one URL to disk atomically: bytes go to a temporary file
 * first, then the temp file is moved into place, so a failed
 * download never leaves a half-written dataset file.
 *
 * @param url         absolute https URL
 * @param destination final path of the downloaded file
 * @throws IOException          non-200 status, empty body, I/O error
 * @throws InterruptedException if interrupted while waiting
 */
private static void downloadFile(String url, Path destination)
        throws IOException, InterruptedException {

    Path temporary = Files.createTempFile(
        destination.getParent(), "fashionmnist-", ".download");

    try {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .timeout(Duration.ofMinutes(3))
            .GET()
            .build();

        HttpResponse<InputStream> response = CLIENT.send(
            request, HttpResponse.BodyHandlers.ofInputStream());

        try (InputStream input = response.body()) {
            if (response.statusCode() != 200) {
                throw new IOException(
                    "Download failed with HTTP status "
                        + response.statusCode() + ": " + url);
            }

            try (OutputStream output =
                     Files.newOutputStream(temporary)) {
                input.transferTo(output);   // efficient bulk copy
            }
        }

        if (Files.size(temporary) == 0) {
            throw new IOException("Downloaded file is empty: " + url);
        }

        Files.move(temporary, destination,
            StandardCopyOption.REPLACE_EXISTING);
    } finally {
        Files.deleteIfExists(temporary);
    }
}
```

### decompress — gzip → raw IDX

```
/**
 * Inflates a .gz file into its raw IDX form.
 *
 * @param gzipFile   the compressed file
 * @param outputFile the raw IDX file to produce
 * @throws IOException if the archive is empty or unreadable
 */
private static void decompress(Path gzipFile, Path outputFile)
        throws IOException {

    Path temporary = Files.createTempFile(
        outputFile.getParent(), "fashionmnist-", ".raw");

    try {
        try (InputStream input =
                 new GZIPInputStream(Files.newInputStream(gzipFile));
             OutputStream output =
                 Files.newOutputStream(temporary)) {
            input.transferTo(output);
        }

        if (Files.size(temporary) == 0) {
            throw new IOException("Decompressed file is empty");
        }

        Files.move(temporary, outputFile,
            StandardCopyOption.REPLACE_EXISTING);
    } finally {
        Files.deleteIfExists(temporary);
    }
}
```

> **Design lesson:** download-and-rename is the standard pattern for dataset fetching. The target filename appears only when the data is complete, so an interrupted run just re-downloads next time.

## 8.2 FashionMNISTLoader

Constants and the nested `Dataset` holder first:

```
public final class FashionMNISTLoader {

    public static final int IMAGE_ROWS = 28;
    public static final int IMAGE_COLS = 28;
    public static final int INPUT_SIZE = IMAGE_ROWS * IMAGE_COLS;  // 784
    public static final int NUM_CLASSES = 10;

    private static final int IMAGE_MAGIC = 2051;   // 0x00000803
    private static final int LABEL_MAGIC = 2049;   // 0x00000801

    /** The ten class names, index = label value. */
    public static final String[] CLASS_NAMES = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    /** Holds normalized images and their integer labels. */
    public static final class Dataset {
        private final double[][] images;   // [N][784], each in [0,1]
        private final int[] labels;        // [N], each in 0..9

        /** @return number of examples in this split */
        public int size()             { return images.length; }

        /** @return all images, [N][784] */
        public double[][] getImages() { return images; }

        /** @param index row index
         *  @return      one 784-pixel vector */
        public double[] getImage(int index) { return images[index]; }

        /** @return all labels, [N] */
        public int[] getLabels()      { return labels; }

        /** @param index row index
         *  @return      class id 0..9 */
        public int getLabel(int index){ return labels[index]; }

        /** @return labels as an [N][10] one-hot matrix */
        public double[][] getOneHotTargets() { /* ... */ }

        /** @param index row index
         *  @return      its one-hot vector */
        public double[] getOneHotTarget(int index) { /* ... */ }
    }
}
```

### Loading entry points

```
/**
 * Loads the 60,000-example training split from a directory.
 *
 * @param directory folder containing the four IDX files
 * @return Dataset with images and labels
 * @throws IOException if a file is missing, malformed or truncated
 */
public static Dataset loadTraining(Path directory) throws IOException {
    return load(
        directory.resolve("train-images-idx3-ubyte"),
        directory.resolve("train-labels-idx1-ubyte"));
}

/**
 * Loads the 10,000-example test split from a directory.
 *
 * @param directory folder containing the four IDX files
 * @return Dataset with images and labels
 * @throws IOException if a file is missing, malformed or truncated
 */
public static Dataset loadTest(Path directory) throws IOException {
    return load(
        directory.resolve("t10k-images-idx3-ubyte"),
        directory.resolve("t10k-labels-idx1-ubyte"));
}

/**
 * Loads and pairs an image file with its label file.
 *
 * @param imageFile IDX file with magic 2051
 * @param labelFile IDX file with magic 2049
 * @return Dataset whose row i pairs image i with label i
 * @throws IOException on header, size or count mismatches
 */
public static Dataset load(Path imageFile, Path labelFile)
        throws IOException {

    double[][] images = loadImages(imageFile);
    int[] labels = loadLabels(labelFile);

    if (images.length != labels.length) {
        throw new IOException(
            "Image count (" + images.length
                + ") does not match label count ("
                + labels.length + ")");
    }

    return new Dataset(images, labels);
}
```

### Parsing the image file

```
/**
 * Reads an IDX image file into a normalized [count][784] array.
 *
 * @param path the raw (decompressed) image file
 * @return pixel matrix with values scaled to [0, 1]
 * @throws IOException wrong magic, wrong dimensions, unexpected
 *                     file size, truncated data, or trailing bytes
 */
public static double[][] loadImages(Path path) throws IOException {
    try (DataInputStream in = input(path)) {
        int magic = in.readInt();
        int count = in.readInt();
        int rows  = in.readInt();
        int cols  = in.readInt();

        if (magic != IMAGE_MAGIC) {
            throw new IOException(
                "Invalid image IDX magic number in " + path);
        }

        if (count <= 0 || rows != IMAGE_ROWS || cols != IMAGE_COLS) {
            throw new IOException(
                "Unexpected Fashion-MNIST image dimensions in " + path);
        }

        // size check before allocating tens of megabytes
        long expectedLength = 16L + (long) count * rows * cols;
        if (Files.size(path) != expectedLength) {
            throw new IOException("Unexpected image file size in " + path);
        }

        double[][] images = new double[count][INPUT_SIZE];

        for (int i = 0; i < count; i++) {
            for (int p = 0; p < INPUT_SIZE; p++) {
                images[i][p] = in.readUnsignedByte() / 255.0;
            }
        }

        ensureEndOfFile(in, path);
        return images;
    } catch (EOFException e) {
        throw new IOException("Truncated image file: " + path, e);
    }
}
```

### Parsing the label file

```
/**
 * Reads an IDX label file into an int array.
 *
 * @param path the raw (decompressed) label file
 * @return label ids, each guaranteed to be in [0, 9]
 * @throws IOException wrong magic, wrong size, bad label value,
 *                     truncated data or trailing bytes
 */
public static int[] loadLabels(Path path) throws IOException {
    try (DataInputStream in = input(path)) {
        int magic = in.readInt();
        int count = in.readInt();

        if (magic != LABEL_MAGIC || count <= 0) {
            throw new IOException("Invalid label IDX header in " + path);
        }

        long expectedLength = 8L + count;
        if (Files.size(path) != expectedLength) {
            throw new IOException("Unexpected label file size in " + path);
        }

        int[] labels = new int[count];

        for (int i = 0; i < count; i++) {
            int label = in.readUnsignedByte();

            if (label >= NUM_CLASSES) {
                throw new IOException(
                    "Invalid label " + label + " at index " + i);
            }

            labels[i] = label;
        }

        ensureEndOfFile(in, path);
        return labels;
    } catch (EOFException e) {
        throw new IOException("Truncated label file: " + path, e);
    }
}
```

### Two small helpers

```
/** Wraps a file stream so readInt/readUnsignedByte work on big-endian bytes. */
private static DataInputStream input(Path path) throws IOException {
    return new DataInputStream(
        new BufferedInputStream(Files.newInputStream(path)));
}

/** Fails the load if any unexpected byte remains after the payload. */
private static void ensureEndOfFile(DataInputStream in, Path path)
        throws IOException {
    if (in.read() != -1) {
        throw new IOException("Unexpected trailing bytes in " + path);
    }
}
```

## 8.3 End-to-end usage

```
Path dir = Path.of("data", "fashion-mnist");

FashionMNISTDownloader.download(dir);              // once

FashionMNISTLoader.Dataset train =
    FashionMNISTLoader.loadTraining(dir);          // 60,000 rows
FashionMNISTLoader.Dataset test =
    FashionMNISTLoader.loadTest(dir);              // 10,000 rows

double[][] x = train.getImages();                  // [60000][784]
double[][] y = train.getOneHotTargets();           // [60000][10]
```

---

[← 07 · Fashion MNIST data](07-data.md) · [09 · Keras-like classifier →](09-keras-like.md)
