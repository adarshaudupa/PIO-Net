#include "stage2_infer.h"
#include "model_data.h"
#include <stdio.h>

static float  last_pred_score = 0.0f;
static float  last_second_score = 0.0f;
static int8_t last_second_class = 0;

float stage2_get_last_score(void) {
    return last_pred_score;
}

int8_t stage2_get_second_class(void) {
    return last_second_class;
}

float stage2_get_second_score(void) {
    return last_second_score;
}

int8_t stage2_classify(const int8_t *in) {
    // Convert int8_t quantized spectrogram (scale 32) to Q8 fixed-point (scale 256)
    static int16_t in_q8[39 * 16];
    for (int i = 0; i < 39 * 16; i++) {
        in_q8[i] = (int16_t)in[i] << 3;
    }

    // 1. Conv1: [1, 39, 16] -> [16, 20, 8] into Arena A
    for (int oc = 0; oc < 16; oc++) {
        const int16_t *w_oc = &conv1_weights[oc * 15];
        int32_t b = conv1_biases[oc];
        for (int oh = 0; oh < 20; oh++) {
            for (int ow = 0; ow < 8; ow++) {
                int32_t acc = b;
                for (int kh = 0; kh < 5; kh++) {
                    int ih = oh * 2 - 2 + kh; // stride 2, pad 2
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw; // stride 2, pad 1
                        if (ih >= 0 && ih < 39 && iw >= 0 && iw < 16) {
                            acc += ((int32_t)in_q8[ih * 16 + iw] * (int32_t)w_oc[kh * 3 + kw]) >> 12;
                        }
                    }
                }
                activation_arena_a[oc * 160 + oh * 8 + ow] = (acc > 0) ? (int16_t)acc : 0;
            }
        }
    }

    // 2. DS1 Depthwise: [16, 20, 8] -> [16, 20, 8] into Arena B
    for (int c = 0; c < 16; c++) {
        const int16_t *w_c = &ds1_dw_weights[c * 9];
        int32_t b = ds1_dw_biases[c];
        for (int h = 0; h < 20; h++) {
            for (int w = 0; w < 8; w++) {
                int32_t acc = b;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = h - 1 + kh; // stride 1, pad 1
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = w - 1 + kw;
                        if (ih >= 0 && ih < 20 && iw >= 0 && iw < 8) {
                            acc += ((int32_t)activation_arena_a[c * 160 + ih * 8 + iw] * (int32_t)w_c[kh * 3 + kw]) >> 12;
                        }
                    }
                }
                activation_arena_b[c * 160 + h * 8 + w] = (acc > 0) ? (int16_t)acc : 0;
            }
        }
    }

    // 3. DS1 Pointwise: [16, 20, 8] -> [24, 20, 8] into Arena A
    for (int oc = 0; oc < 24; oc++) {
        const int16_t *w_oc = &ds1_pw_weights[oc * 16];
        int32_t b = ds1_pw_biases[oc];
        for (int i = 0; i < 160; i++) {
            int32_t acc = b;
            for (int ic = 0; ic < 16; ic++) {
                acc += ((int32_t)activation_arena_b[ic * 160 + i] * (int32_t)w_oc[ic]) >> 12;
            }
            activation_arena_a[oc * 160 + i] = (acc > 0) ? (int16_t)acc : 0;
        }
    }

    // 4. DS2 Depthwise: [24, 20, 8] -> [24, 10, 4] into Arena B
    for (int c = 0; c < 24; c++) {
        const int16_t *w_c = &ds2_dw_weights[c * 9];
        int32_t b = ds2_dw_biases[c];
        for (int oh = 0; oh < 10; oh++) {
            for (int ow = 0; ow < 4; ow++) {
                int32_t acc = b;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh * 2 - 1 + kh; // stride 2, pad 1
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw;
                        if (ih >= 0 && ih < 20 && iw >= 0 && iw < 8) {
                            acc += ((int32_t)activation_arena_a[c * 160 + ih * 8 + iw] * (int32_t)w_c[kh * 3 + kw]) >> 12;
                        }
                    }
                }
                activation_arena_b[c * 40 + oh * 4 + ow] = (acc > 0) ? (int16_t)acc : 0;
            }
        }
    }

    // 5. DS2 Pointwise: [24, 10, 4] -> [32, 10, 4] into Arena A
    for (int oc = 0; oc < 32; oc++) {
        const int16_t *w_oc = &ds2_pw_weights[oc * 24];
        int32_t b = ds2_pw_biases[oc];
        for (int i = 0; i < 40; i++) {
            int32_t acc = b;
            for (int ic = 0; ic < 24; ic++) {
                acc += ((int32_t)activation_arena_b[ic * 40 + i] * (int32_t)w_oc[ic]) >> 12;
            }
            activation_arena_a[oc * 40 + i] = (acc > 0) ? (int16_t)acc : 0;
        }
    }

    // 6. DS3 Depthwise: [32, 10, 4] -> [32, 10, 4] into Arena B
    for (int c = 0; c < 32; c++) {
        const int16_t *w_c = &ds3_dw_weights[c * 9];
        int32_t b = ds3_dw_biases[c];
        for (int h = 0; h < 10; h++) {
            for (int w = 0; w < 4; w++) {
                int32_t acc = b;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = h - 1 + kh; // stride 1, pad 1
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = w - 1 + kw;
                        if (ih >= 0 && ih < 10 && iw >= 0 && iw < 4) {
                            acc += ((int32_t)activation_arena_a[c * 40 + ih * 4 + iw] * (int32_t)w_c[kh * 3 + kw]) >> 12;
                        }
                    }
                }
                activation_arena_b[c * 40 + h * 4 + w] = (acc > 0) ? (int16_t)acc : 0;
            }
        }
    }

    // 7. DS3 Pointwise: [32, 10, 4] -> [32, 10, 4] into Arena A
    for (int oc = 0; oc < 32; oc++) {
        const int16_t *w_oc = &ds3_pw_weights[oc * 32];
        int32_t b = ds3_pw_biases[oc];
        for (int i = 0; i < 40; i++) {
            int32_t acc = b;
            for (int ic = 0; ic < 32; ic++) {
                acc += ((int32_t)activation_arena_b[ic * 40 + i] * (int32_t)w_oc[ic]) >> 12;
            }
            activation_arena_a[oc * 40 + i] = (acc > 0) ? (int16_t)acc : 0;
        }
    }

    // 8. Global Average Pooling: [32, 10, 4] -> pool[32]
    int32_t pooled[32];
    for (int c = 0; c < 32; c++) {
        int32_t sum = 0;
        for (int i = 0; i < 40; i++) {
            sum += activation_arena_a[c * 40 + i];
        }
        pooled[c] = sum / 40;
    }

    // 9. Fully Connected Classifier: [32] -> 12 class logits
    int8_t best_class = 0;
    int32_t max_logit = -2147483647;
    int8_t second_class = 0;
    int32_t second_logit = -2147483647;

    for (int cl = 0; cl < 12; cl++) {
        int32_t acc = fc_biases[cl];
        const int16_t *w_cl = &fc_weights[cl * 32];
        for (int i = 0; i < 32; i++) {
            acc += (pooled[i] * (int32_t)w_cl[i]) >> 12;
        }
        if (acc > max_logit) {
            second_logit = max_logit;
            second_class = best_class;
            max_logit = acc;
            best_class = (int8_t)cl;
        } else if (acc > second_logit) {
            second_logit = acc;
            second_class = (int8_t)cl;
        }
    }

    last_pred_score = (float)max_logit / 256.0f;
    last_second_score = (float)second_logit / 256.0f;
    last_second_class = second_class;
    return best_class;
}

int8_t stage2_classify_float(const float *in) {
    static int8_t in_q7[39 * 16];
    for (int i = 0; i < 39 * 16; i++) {
        int32_t val = (int32_t)(in[i] * 32.0f);
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        in_q7[i] = (int8_t)val;
    }
    return stage2_classify(in_q7);
}