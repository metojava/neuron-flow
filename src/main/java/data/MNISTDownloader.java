package data;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Downloads the four raw MNIST IDX files (train images,
 * train labels, t10k images, t10k labels) from a base URL into
 * a local directory, skipping files that already exist.
 * Works for both the original MNIST and Fashion-MNIST mirrors
 * since only the base URL differs.
 */
public class MNISTDownloader {

    /** File names expected on the server and on disk. */
    private static final String[] FILES = {
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    };

    private final String baseUrl;
    private final HttpClient client;

    /**
     * Creates a downloader for a given base URL.
     *
     * @param baseUrl directory URL ending with '/', e.g.
     *                {@code https://.../data/fashion/}
     */
    public MNISTDownloader(String baseUrl) {
        if (baseUrl == null || baseUrl.isEmpty()) {
            throw new IllegalArgumentException("baseUrl required");
        }
        this.baseUrl = baseUrl.endsWith("/")
            ? baseUrl : baseUrl + "/";
        this.client = HttpClient.newBuilder()
            .followRedirects(HttpClient.Redirect.NORMAL)
            .build();
    }

    /**
     * Downloads every IDX file that is not already present.
     *
     * @param directory target directory; created if missing
     * @return the directory path
     * @throws IOException on network or filesystem failure
     */
    public Path download(Path directory) throws IOException {
        Files.createDirectories(directory);

        for (String name : FILES) {
            Path target = directory.resolve(name);

            if (Files.exists(target)) {
                System.out.println("Already available: " + target);
                continue;
            }

            System.out.println("Downloading " + name + " ...");
            fetch(baseUrl + name, target);
            System.out.println("  Saved to " + target);
        }

        return directory;
    }

    /**
     * Downloads one file, writing to a temporary sibling first
     * then atomically moving it into place.
     */
    private void fetch(String url, Path target) throws IOException {
        Path temp = target.resolveSibling(
            target.getFileName() + ".part");

        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .GET()
                .build();

            HttpResponse<InputStream> response =
                client.send(request,
                    HttpResponse.BodyHandlers.ofInputStream());

            if (response.statusCode() != 200) {
                throw new IOException(
                    "HTTP " + response.statusCode() + " for " + url);
            }

            try (InputStream in = response.body()) {
                Files.copy(in, temp,
                    StandardCopyOption.REPLACE_EXISTING);
            }

            Files.move(temp, target,
                StandardCopyOption.REPLACE_EXISTING);

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Download interrupted", e);
        } finally {
            Files.deleteIfExists(temp);
        }
    }
}
