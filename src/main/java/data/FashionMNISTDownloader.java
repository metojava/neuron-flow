package data;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Downloads the four Fashion-MNIST IDX archives when needed. */
public final class FashionMNISTDownloader {
    private static final String BASE_URL =
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/";

    private FashionMNISTDownloader() {
    }

    public static Path ensureDownloaded() throws IOException {
        return ensureDownloaded(Paths.get("data", "fashion-mnist"));
    }

    public static Path ensureDownloaded(Path directory) throws IOException {
        return new MNISTDownloader(BASE_URL).download(directory);
    }
}