#include "stage2_infer.h"
#include "model_data.h"
#include <stdio.h>

// Fixed-point scaling: accumulator * multiplier >> (31 + shift)
static inline int8_t scale_relu(int32_t acc, int32_t mult, int32_t shift) {
    int64_t prod = (int64_t)acc * mult;
    int64_t scaled = (prod + (1LL << 30)) >> 31;
    int32_t res = (int32_t)(scaled >> shift);
    if (res < 0) res = 0;       // ReLU
    if (res > 127) res = 127;   // int8 clamp
    return (int8_t)res;
}

int8_t stage2_classify(const int8_t *in) {
    // 1. Conv1: [1, 39, 16] -> [16, 20, 8] into Arena A
    for (int oc = 0; oc < 16; oc++) {
        const int8_t *w_oc = &conv1_weights[oc * 15]; // 5x3 kernel
        for (int oh = 0; oh < 20; oh++) {
            for (int ow = 0; ow < 8; ow++) {
                int32_t acc = conv1_biases[oc];
                for (int kh = 0; kh < 5; kh++) {
                    int ih = oh * 2 - 2 + kh; // stride 2, pad 2
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw; // stride 2, pad 1
                        if (ih >= 0 && ih < 39 && iw >= 0 && iw < 16) {
                            acc += (int32_t)in[ih * 16 + iw] * (int32_t)w_oc[kh * 3 + kw];
                        }
                    }
                }
                activation_arena_a[oc * 160 + oh * 8 + ow] = scale_relu(acc, conv1_mult[oc], conv1_shift[oc]);
            }
        }
    }

    // 2. DS1 Depthwise: [16, 20, 8] -> [16, 20, 8] into Arena B
    for (int c = 0; c < 16; c++) {
        const int8_t *w_c = &ds1_dw_weights[c * 9];
        for (int h = 0; h < 20; h++) {
            for (int w = 0; w < 8; w++) {
                int32_t acc = ds1_dw_biases[c];
                for (int kh = 0; kh < 3; kh++) {
                    int ih = h - 1 + kh; // stride 1, pad 1
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = w - 1 + kw;
                        if (ih >= 0 && ih < 20 && iw >= 0 && iw < 8) {
                            acc += (int32_t)activation_arena_a[c * 160 + ih * 8 + iw] * (int32_t)w_c[kh * 3 + kw];
                        }
                    }
                }
                activation_arena_b[c * 160 + h * 8 + w] = scale_relu(acc, ds1_dw_mult[c], ds1_dw_shift[c]);
            }
        }
    }

    // 3. DS1 Pointwise: [16, 20, 8] -> [24, 20, 8] into Arena A
    for (int oc = 0; oc < 24; oc++) {
        const int8_t *w_oc = &ds1_pw_weights[oc * 16];
        for (int i = 0; i < 160; i++) { // 20 * 8 spatial elements
            int32_t acc = ds1_pw_biases[oc];
            for (int ic = 0; ic < 16; ic++) {
                acc += (int32_t)activation_arena_b[ic * 160 + i] * (int32_t)w_oc[ic];
            }
            activation_arena_a[oc * 160 + i] = scale_relu(acc, ds1_pw_mult[oc], ds1_pw_shift[oc]);
        }
    }

    // 4. DS2 Depthwise: [24, 20, 8] -> [24, 10, 4] into Arena B
    for (int c = 0; c < 24; c++) {
        const int8_t *w_c = &ds2_dw_weights[c * 9];
        for (int oh = 0; oh < 10; oh++) {
            for (int ow = 0; ow < 4; ow++) {
                int32_t acc = ds2_dw_biases[c];
                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh * 2 - 1 + kh; // stride 2, pad 1
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 - 1 + kw;
                        if (ih >= 0 && ih < 20 && iw >= 0 && iw < 8) {
                            acc += (int32_t)activation_arena_a[c * 160 + ih * 8 + iw] * (int32_t)w_c[kh * 3 + kw];
                        }
                    }
                }
                activation_arena_b[c * 40 + oh * 4 + ow] = scale_relu(acc, ds2_dw_mult[c], ds2_dw_shift[c]);
            }
        }
    }

    // 5. DS2 Pointwise: [24, 10, 4] -> [32, 10, 4] into Arena A
    for (int oc = 0; oc < 32; oc++) {
        const int8_t *w_oc = &ds2_pw_weights[oc * 24];
        for (int i = 0; i < 40; i++) { // 10 * 4 spatial elements
            int32_t acc = ds2_pw_biases[oc];
            for (int ic = 0; ic < 24; ic++) {
                acc += (int32_t)activation_arena_b[ic * 40 + i] * (int32_t)w_oc[ic];
            }
            activation_arena_a[oc * 40 + i] = scale_relu(acc, ds2_pw_mult[oc], ds2_pw_shift[oc]);
        }
    }

    // 6. DS3 Depthwise: [32, 10, 4] -> [32, 10, 4] into Arena B
    for (int c = 0; c < 32; c++) {
        const int8_t *w_c = &ds3_dw_weights[c * 9];
        for (int h = 0; h < 10; h++) {
            for (int w = 0; w < 4; w++) {
                int32_t acc = ds3_dw_biases[c];
                for (int kh = 0; kh < 3; kh++) {
                    int ih = h - 1 + kh; // stride 1, pad 1
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = w - 1 + kw;
                        if (ih >= 0 && ih < 10 && iw >= 0 && iw < 4) {
                            acc += (int32_t)activation_arena_a[c * 40 + ih * 4 + iw] * (int32_t)w_c[kh * 3 + kw];
                        }
                    }
                }
                activation_arena_b[c * 40 + h * 4 + w] = scale_relu(acc, ds3_dw_mult[c], ds3_dw_shift[c]);
            }
        }
    }

    // 7. DS3 Pointwise: [32, 10, 4] -> [32, 10, 4] into Arena A
    for (int oc = 0; oc < 32; oc++) {
        const int8_t *w_oc = &ds3_pw_weights[oc * 32];
        for (int i = 0; i < 40; i++) {
            int32_t acc = ds3_pw_biases[oc];
            for (int ic = 0; ic < 32; ic++) {
                acc += (int32_t)activation_arena_b[ic * 40 + i] * (int32_t)w_oc[ic];
            }
            activation_arena_a[oc * 40 + i] = scale_relu(acc, ds3_pw_mult[oc], ds3_pw_shift[oc]);
        }
    }

    // 8. Global Average Pooling: [32, 10, 4] -> pool[32]
    int8_t pooled[32];
    for (int c = 0; c < 32; c++) {
        int32_t sum = 0;
        for (int i = 0; i < 40; i++) {
            sum += activation_arena_a[c * 40 + i];
        }
        pooled[c] = (int8_t)(sum / 40);
    }

    // 9. Fully Connected Classifier: [32] -> 12 class logits
    int8_t best_class = 0;
    int32_t max_logit = -2147483647;
    //printf("[C-Logits] ");
    for (int cl = 0; cl < 12; cl++) {
        int32_t acc = fc_biases[cl];
        const int8_t *w_cl = &fc_weights[cl * 32];
        for (int i = 0; i < 32; i++) {
            acc += (int32_t)pooled[i] * (int32_t)w_cl[i];
        }
        int64_t prod = (int64_t)acc * fc_mult[cl];
        int64_t scaled = (prod + (1LL << 30)) >> 31;
        int32_t logit = (int32_t)(scaled >> fc_shift[cl]);
       // printf("%d:%ld ", cl, (long)logit);
        if (logit > max_logit) {
            max_logit = logit;
            best_class = (int8_t)cl;
        }
    }

    return best_class;
}