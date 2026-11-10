# Info
I created this to compare the sizes of TFs and target genes when:

## a) using a TF list from TAIR
B=1024, F=56, G=944, k=15, w=2, d=3, n=1; memory in GiB

peak                 total
dense               88.298
sparse              27.037
compile              1.704

## b) not using a TF list from TAIR
B=1024, F=422, G=578, k=15, w=2, d=3, n=1; memory in GiB

peak                 total
dense               33.822
sparse              16.714
compile              1.209
