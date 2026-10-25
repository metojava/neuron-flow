package data.model;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

/**
 * Sequential and positional reads from a file channel, with bounds checks.
 * Used by the lazy model readers so that only headers and requested tensors
 * are read from disk.
 */
final class ChannelCursor {

    private final FileChannel channel;
    private final long size;
    private long pos;

    ChannelCursor(FileChannel channel) throws IOException {
        this.channel = channel;
        this.size = channel.size();
    }

    long position() {
        return pos;
    }

    long size() {
        return size;
    }

    /** Moves the cursor to an absolute position. */
    void seek(long p) {
        pos = p;
    }

    /** Reads {@code n} bytes at the cursor and advances it. */
    byte[] read(int n) throws IOException {
        byte[] b = readAt(pos, n);
        pos += n;
        return b;
    }

    /** Reads {@code n} little-endian bytes at the cursor and advances it. */
    ByteBuffer le(int n) throws IOException {
        return ByteBuffer.wrap(read(n)).order(ByteOrder.LITTLE_ENDIAN);
    }

    /** Reads {@code n} bytes starting at {@code at}; the cursor is not moved. */
    byte[] readAt(long at, long n) throws IOException {
        if (n < 0 || n > Integer.MAX_VALUE || at < 0 || at + n > size) {
            throw new IOException("Unexpected end of file (need " + n + " bytes at " + at + ")");
        }
        byte[] out = new byte[(int) n];
        ByteBuffer bb = ByteBuffer.wrap(out);
        long p = at;
        while (bb.hasRemaining()) {
            int r = channel.read(bb, p);
            if (r < 0) throw new IOException("Unexpected end of file");
            p += r;
        }
        return out;
    }
}
