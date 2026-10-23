#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "d:/pico/pionet_core/stage2_features.h"

int main(int argc, char **argv) {
    static int16_t pcm[16000];
    static float spec[39 * 16];
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    fread(pcm, 2, 16000, f);
    fclose(f);
    stage2_features_init();
    stage2_extract_spectrogram(pcm, spec);
    f = fopen(argv[2], "wb");
    if (!f) return 1;
    fwrite(spec, 4, 39 * 16, f);
    fclose(f);
    return 0;
}
