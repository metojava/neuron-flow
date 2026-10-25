package org.example.deeplearning.data;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.zip.GZIPInputStream;

public final class MNISTDownloader {

    private static final String BASE_URL =
        "https://storage.googleapis.com/cvdf-datasets/mnist/";

    private static final String[] FILES = {
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte"
    };

    private static final HttpClient CLIENT = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(30))
        .followRedirects(HttpClient.Redirect.NORMAL)
        .build();

    private MNISTDownloader() {
    }

    public static Path download() throws IOException, InterruptedException {
        return download(Path.of("data", "mnist"));
    }

    public static Path download(Path directory)
            throws IOException, InterruptedException {

        Files.createDirectories(directory);

        for (String file : FILES) {
            Path rawFile = directory.resolve(file);
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
                Files.deleteIfExists(gzipFile);
            }

            System.out.println("Saved " + rawFile);
        }

        System.out.println("MNIST files are ready in: "
            + directory.toAbsolutePath());

        return directory;
    }

    private static void downloadFile(String url, Path destination)
            throws IOException, InterruptedException {

        Path temporary = Files.createTempFile(
            destination.getParent(), "mnist-", ".download");

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

                try (OutputStream output = Files.newOutputStream(temporary)) {
                    input.transferTo(output);
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

    private static void decompress(Path gzipFile, Path outputFile)
            throws IOException {

        Path temporary = Files.createTempFile(
            outputFile.getParent(), "mnist-", ".raw");

        try {
            try (InputStream input = new GZIPInputStream(
                     Files.newInputStream(gzipFile));
                 OutputStream output = Files.newOutputStream(temporary)) {
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

    public static void main(String[] args) throws Exception {
        Path directory = args.length > 0
            ? Path.of(args[0])
            : Path.of("data", "mnist");

        download(directory);
    }
}