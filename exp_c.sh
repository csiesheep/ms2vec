#!/bin/bash

graph=$1
folder=$2

#graph='/export/home/txf225/ms2vec/dblp/PA_20V_94_14.graph'
#folder='/export/home/txf225/ms2vec/dblp/classification/'

postfix=''
#for l in 1280
for l in 1280
do
    for d in 128
    do
        for w in 3 2 1
        do
            for i in 4
            do
                for e in ''
                do
                    for s in ''
                    do
                        for r in 0.5
                        do
                            for n in 5
                            do
                                for m in ''
                                do
                                    fname='c_n2v_l'$l'_w'$w'_d'$d'_i'$i'_'$e'_'$s'_r'$r'_n'$n'_'$m$postfix'.txt'
                                    rfname='c_r2v_l'$l'_w'$w'_d'$d'_i'$i'_'$e'_'$s'_r'$r'_n'$n'_'$m$postfix'.txt'
                                    gfname='c_g2v_l'$l'_w'$w'_d'$d'_i'$i'_'$e'_'$s'_r'$r'_n'$n'_'$m$postfix'.txt'
                                    f='c_data_l'$l'_w'$w'.txt'
                                    q='c_freq.txt'
                                    g='c_matcher_w'$w'.txt'
                                    echo $fname
                                    python main_c.py $graph $folder/$fname $folder/$rfname $folder/$gfname -d $d -l $l -p 16 -w $w -i $i $e $s -r $r -n $n $m -f $folder/$f -q $folder/$q -g $folder/$g
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
