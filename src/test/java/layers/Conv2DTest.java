package layers;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Conv2DTest {

    private static final double H = 1e-3;

    /** Fixed linear readout so loss = sum(out * readout). */
    private static float[][][] readout(int c, int h, int w) {
        float[][][] r = new float[c][h][w];
        int n = 0;
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < h; j++) {
                for (int k = 0; k < w; k++) {
                    r[i][j][k] = (float) Math.sin(0.37 * n++);
                }
            }
        }
        return r;
    }

    private static double loss(Conv2D conv, float[][][] input,
                               float[][][] r) {
        float[][][] out = conv.forward(input);
        double sum = 0.0;
        for (int c = 0; c < out.length; c++) {
            for (int i = 0; i < out[c].length; i++) {
                for (int j = 0; j < out[c][i].length; j++) {
                    sum += out[c][i][j] * r[c][i][j];
                }
            }
        }
        return sum;
    }

    @Test
    void multiChannelGradientsMatchFiniteDifferences() {
        Conv2D conv = new Conv2D(2, 3, 3, 3, 1, 1, 5L);
        float[][][] input = new float[2][5][5];
        for (int c = 0; c < 2; c++) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    input[c][i][j] = (float) Math.sin(c * 7 + i * 1.3 + j * 0.7);
                }
            }
        }
        float[][][] out = conv.forward(input);
        float[][][] r = readout(out.length, out[0].length, out[0][0].length);

        conv.zeroGrad();
        float[][][] dIn = conv.backward(r);

        // Input gradient.
        for (int c = 0; c < 2; c++) {
            for (int i = 0; i < 5; i += 2) {
                for (int j = 0; j < 5; j += 2) {
                    float saved = input[c][i][j];
                    input[c][i][j] = (float) (saved + H);
                    double up = loss(conv, input, r);
                    input[c][i][j] = (float) (saved - H);
                    double down = loss(conv, input, r);
                    input[c][i][j] = saved;
                    assertEquals((up - down) / (2 * H), dIn[c][i][j], 2e-3,
                        "input grad at " + c + "," + i + "," + j);
                }
            }
        }

        // Weight gradient: finite difference at the current weight, then
        // with lr=1 and batch=1 the update subtracts the analytic gradient.
        float saved = conv.weights[1][0][2][1];
        conv.weights[1][0][2][1] = (float) (saved + H);
        double up = loss(conv, input, r);
        conv.weights[1][0][2][1] = (float) (saved - H);
        double down = loss(conv, input, r);
        conv.weights[1][0][2][1] = saved;
        double numeric = (up - down) / (2 * H);

        conv.update(1.0f, 1);
        float analytic = saved - conv.weights[1][0][2][1];
        assertEquals(numeric, analytic, 2e-3, "weight grad");
    }

    /** Naive reference convolution, independent of the layer's code path. */
    private static float[][][] referenceForward(Conv2D conv, float[][][] in,
                                                int stride, int pad,
                                                int outC, int inC, int k,
                                                int outH, int outW) {
        float[][][] out = new float[outC][outH][outW];
        for (int oc = 0; oc < outC; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    double s = conv.biases[oc];
                    for (int ic = 0; ic < inC; ic++) {
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = oh * stride - pad + kh;
                                int iw = ow * stride - pad + kw;
                                if (ih >= 0 && ih < in[0].length
                                    && iw >= 0 && iw < in[0][0].length) {
                                    s += in[ic][ih][iw] * conv.weights[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    out[oc][oh][ow] = (float) s;
                }
            }
        }
        return out;
    }

    @Test
    void stride1VectorPathMatchesReferenceForSeveralShapes() {
        int[][] cases = {
            // inH, inW, pad, kernel
            {7, 9, 0, 3}, {7, 9, 1, 3}, {6, 5, 2, 5}, {28, 28, 1, 3}
        };
        for (int[] c : cases) {
            int inH = c[0], inW = c[1], pad = c[2], k = c[3];
            Conv2D conv = new Conv2D(2, 3, k, k, 1, pad, 11L);
            float[][][] in = new float[2][inH][inW];
            for (int ch = 0; ch < 2; ch++) {
                for (int i = 0; i < inH; i++) {
                    for (int j = 0; j < inW; j++) {
                        in[ch][i][j] = (float) Math.cos(ch + 0.5 * i - 0.3 * j);
                    }
                }
            }
            int outH = conv.outputSize(inH);
            int outW = conv.outputSize(inW);
            float[][][] got = conv.forward(in);
            float[][][] want = referenceForward(conv, in, 1, pad, 3, 2, k, outH, outW);
            for (int oc = 0; oc < 3; oc++) {
                for (int i = 0; i < outH; i++) {
                    assertArrayEquals(want[oc][i], got[oc][i], 1e-4f,
                        "shape " + inH + "x" + inW + " pad " + pad + " k " + k);
                }
            }
        }
    }

    @Test
    void strideTwoGradientsMatchFiniteDifferences() {
        Conv2D conv = new Conv2D(2, 2, 3, 3, 2, 1, 4L);
        float[][][] input = new float[2][7][7];
        for (int c = 0; c < 2; c++) {
            for (int i = 0; i < 7; i++) {
                for (int j = 0; j < 7; j++) {
                    input[c][i][j] = (float) Math.sin(c + i * 0.9 + j * 0.4);
                }
            }
        }
        float[][][] out = conv.forward(input);
        float[][][] r = readout(out.length, out[0].length, out[0][0].length);
        conv.zeroGrad();
        float[][][] dIn = conv.backward(r);
        for (int c = 0; c < 2; c++) {
            for (int i = 0; i < 7; i += 3) {
                for (int j = 0; j < 7; j += 3) {
                    float saved = input[c][i][j];
                    input[c][i][j] = (float) (saved + H);
                    double up = loss(conv, input, r);
                    input[c][i][j] = (float) (saved - H);
                    double down = loss(conv, input, r);
                    input[c][i][j] = saved;
                    assertEquals((up - down) / (2 * H), dIn[c][i][j], 2e-3);
                }
            }
        }
    }

    @Test
    void rejectsWrongChannelCountAndBackwardBeforeForward() {
        Conv2D conv = new Conv2D(2, 3, 3, 3, 1, 1, 5L);
        assertThrows(IllegalStateException.class,
            () -> conv.backward(new float[3][5][5]));
        assertThrows(IllegalArgumentException.class,
            () -> conv.forward(new float[1][5][5]));
    }

    @Test
    void parameterCountCoversAllWeightsAndBiases() {
        Conv2D conv = new Conv2D(4, 8, 3, 3, 1, 1, 1L);
        assertEquals(8 * 4 * 3 * 3 + 8, conv.parameterCount());
    }
}
