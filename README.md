# HyprEQ
What this setup does: a 256-tap-per-phase, 4,096-bank Kaiser–Bessel ( β = 16 β=16 , -165 dB) polyphase sinc engine into a 65,536-tap (262,144-pt FFT, 0.183 Hz resolution) 64-bit IEEE double Minimum-Phase Convolver
af=lavfi=[aformat=sample_fmts=dblp,aresample=48000:filter_size=256:phase_shift=12:cutoff=0.98:filter_type=kaiser:kaiser_beta=16:osf=dblp[in48];amovie=/etc/mpv/ir.wav,aformat=sample_fmts=dblp[ir2];[in48][ir2]afir=irfmt=mono:irnorm=-1:wet=1:minp=4096:maxp=4096:precision=double]
