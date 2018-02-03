#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define MAX_RW_LENGTH 10000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30M nodes in the vocabulary
const int mp_vocab_hash_size = 1000;

typedef float real;                    // Precision of float numbers

char train_file[MAX_STRING], output_file[MAX_STRING], role_output_file[MAX_STRING], graphlet_output_file[MAX_STRING], freq_file[MAX_STRING];
int binary = 0, debug_mode = 2, window = 3, num_threads = 1;
int sigmoid_reg = 0;
long long node_count = 0, layer1_size = 100, role_count = 0, graphlet_count = 0;
long long total_data_count = 0, data_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, role_ratio=1.0, g_ratio = 0.0;
real *syn0, *synr, *syng, *expTable;
clock_t start;
int hs = 1, negative = 5;
const int table_size = 1e8;
int *table;
int iteration=1, equal=0;

void InitUnigramTable() {
  int freq = 0, a;
  int freqs[node_count];
  int id = 0;
  real total = 0;
  real d1, power = 0.75;
  FILE *fin;

  printf("Starting learning frequencies using file %s\n", freq_file);
  table = (int *)malloc(table_size * sizeof(int));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  fin = fopen(freq_file, "rb");
  if (fin == NULL) {
    printf("ERROR: node frequence file not found!\n");
    exit(1);
  } 

  fscanf(fin, "%d", &freq);    
  id = 0;
  while (!feof (fin))
  { 
    total += pow(freq, power);
    freqs[id] = freq;
    id++;
//  printf("%d ", freq);
    fscanf(fin, "%d", &freq);      
  }
  fclose(fin); 
//printf("total:%f\n", total);    

  id = 0;
  d1 = pow(freqs[id], power) / (real)total;
  for (a = 0; a < table_size; a++) {
    table[a] = id;
    if (a / (real)table_size > d1) {
//    printf("id:%d p:%f a:%d\n", id, d1, a);    
      id++;
      d1 += pow(freqs[id], power) / (real)total;
    }
    if (id >= node_count) id = node_count- 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t')) {
      if (a > 0) {
        break;
      }
      continue;
    }
    if (ch == '\n') {
        if (a > 0) {
            ungetc(ch, fin);
            break;
        }
        strcpy(word, (char *)"\n");
        return;
    }
    word[a] = ch;
    a++;
  }
  word[a] = 0;
}

void InitNet() {
  long long a, b;
  a = posix_memalign((void **)&syn0, 128, (long long)node_count * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (b = 0; b < layer1_size; b++) for (a = 0; a < node_count; a++)
    syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;

  a = posix_memalign((void **)&synr, 128, (long long)role_count* layer1_size * sizeof(real));
  if (synr== NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (b = 0; b < layer1_size; b++) for (a = 0; a < role_count; a++)
    synr[a * layer1_size + b] = (rand() / (real)RAND_MAX) / layer1_size;

  a = posix_memalign((void **)&syng, 128, (long long)graphlet_count* layer1_size * sizeof(real));
  if (syng== NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (b = 0; b < layer1_size; b++) for (a = 0; a < graphlet_count; a++)
    syng[a * layer1_size + b] = (rand() / (real)RAND_MAX) / layer1_size;
}

void DestroyNet() {
}

void *TrainModelThread(void *id) {
  long long a, b;
  char item[MAX_STRING];
  long long data_count = 0, last_data_count = 0;
  real f, g;
  clock_t now;
  real sigmoid = 0, sigmoid2 = 0;
  unsigned long long next_random = (long long)id;
  real *ex = (real *)calloc(layer1_size, sizeof(real));
  real *exr = (real *)calloc(layer1_size, sizeof(real));
  real *eyr = (real *)calloc(layer1_size, sizeof(real));
  real *eg = (real *)calloc(layer1_size, sizeof(real));
  real *xy = (real *)calloc(layer1_size, sizeof(real));
  long long lx, ly, lxr, lyr, lg, c, label;
  char tokens[5][MAX_STRING];
  char *token;
  int token_length = 0;
  int x_size = 0, xr_size = 0;
  int xs[window];
  int xrs[window];
  int x, xr, y, yr, pos_y, gid;
  char* pSave = NULL;
  int iter=0;
  FILE *fi = NULL;
  float weight = 1.0;

  for (iter=0; iter<iteration; iter++)
  {
    fi = fopen(train_file, "rb");
    if (fi == NULL) {
      fprintf(stderr, "no such file or directory: %s", train_file);
      exit(1);
    }
    fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
    while (1)
    {
      //Printing learning progress
      if (data_count - last_data_count > 100) {
        data_count_actual += data_count - last_data_count;
        last_data_count = data_count;
        if ((debug_mode > 1)) {
          now=clock();
          printf("%cAlpha: %f  Progress(%lld/%lld): %.2f%%  data/thread/sec: %.2fk", 13, alpha,
           data_count_actual, total_data_count,
           data_count_actual / (real)(total_data_count + 1) * 100,
           data_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
          fflush(stdout);
        }
        alpha = starting_alpha * (1 - data_count_actual / (real)(total_data_count + 1));
        if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
      }
     
      //Loading a data
      token_length = 0;
      if (feof(fi) == 1) break;
      while (1) {
        ReadWord(item, fi);
        if (feof(fi) == 1) break;
        if (strcmp(item, "\n") == 0) {
            break;
        }
        strcpy(tokens[token_length], item);
        token_length++;
      }
      if (token_length != 5) continue;
     
      //Parsing
      gid = atoi(tokens[0]);
      pos_y = atoi(tokens[1]);
      yr = atoi(tokens[2]);
//    printf("y:%d yr:%d ", y, yr);
      token = strtok_r(tokens[3], ",", &pSave);
      x_size = 0;
      while (token != NULL)
      {
        xs[x_size] = atoi(token);
//      printf("x%d:%d ", x_size, xs[x_size]);
        x_size++;
        token = strtok_r(NULL, ",", &pSave);
      }
      token = strtok_r(tokens[4], ",", &pSave);
      xr_size = 0;
      while (token != NULL)
      {
        xrs[xr_size] = atoi(token);
//      printf("xr%d:%d ", xr_size, xrs[xr_size]);
        xr_size++;
        token = strtok_r(NULL, ",", &pSave);
      }
//    printf("\n");
      if (feof(fi)) break;
      if (data_count >= total_data_count / num_threads) break;
      data_count++;

      lyr = yr * layer1_size;
      lg = gid * layer1_size;
     
      //Learning
      if (equal == 1) weight = 1.0/x_size;
      for (a=0; a<x_size; a++)
      {
        x = xs[a];
        xr = xrs[a];
        lx = x * layer1_size;
     
        next_random = next_random * (unsigned long long)25214903917 + 11;
        for (b = 0; b < negative + 1; b++) {
          if (b == 0) {
            label = 1;
            y = pos_y;
          } else {
            // negative sampling
            next_random = next_random * (unsigned long long)25214903917 + 11;
            y = table[(next_random >> 16) % table_size];
            if (y == x || y == pos_y) continue;
            label = 0;
          }
     
          ly = y * layer1_size;

          //----------------------------
          //Training of a data for roles
          //----------------------------
          if (role_ratio != 0.0)
          {
            lxr = xr * layer1_size;
            for (c = 0; c < layer1_size; c++)
            {
              ex[c] = 0;
              exr[c] = 0;
              eyr[c] = 0;
              xy[c] = syn0[c + lx] * syn0[c + ly];
            }
            
            //Predicting
            f = 0;
            for (c = 0; c < layer1_size; c++) {
              if (sigmoid_reg) {
                if (synr[c + lxr] < -MAX_EXP || synr[c + lyr] < -MAX_EXP) continue;
                if (synr[c + lxr] > MAX_EXP) {
                  if (synr[c + lyr] > MAX_EXP) f += xy[c];
                  else {
                    sigmoid = expTable[(int)((synr[c + lyr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                    f += sigmoid * xy[c];
                  }
                }
                else if (synr[c + lyr] > MAX_EXP) {
                  sigmoid = expTable[(int)((synr[c + lxr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                  f += sigmoid * xy[c];
                }
                else {
                  sigmoid = expTable[(int)((synr[c + lxr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                  sigmoid2 = expTable[(int)((synr[c + lyr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                  f += sigmoid * sigmoid2 * xy[c];
                }
              } else {
                if (synr[c + lxr] >= 0 && synr[c + lyr] >= 0)
                {
                  f += xy[c];
                }
              }
            }
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            g = g * weight * role_ratio;
           
            //Updating
            //error of xr
            for (c = 0; c < layer1_size; c++) {
              if (sigmoid_reg)
              {
                if (synr[c + lyr] < -MAX_EXP) continue; 
                else if (synr[c + lyr] > MAX_EXP)
                {
                  f = synr[c + lxr];
                  if (f > MAX_EXP || f < -MAX_EXP) continue;
                  sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                  exr[c] = g * xy[c] * sigmoid * (1-sigmoid);
                }
                else
                {
                  sigmoid2 = expTable[(int)((synr[c + lyr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                  f = synr[c + lxr];
                  if (f > MAX_EXP || f < -MAX_EXP) continue;
                  sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                  exr[c] = g * xy[c] * sigmoid * (1-sigmoid) * sigmoid2;
                }
              }
              else
              {
                if (synr[c + lyr] < 0) continue; 
                f = synr[c + lxr];
                if (f > MAX_EXP || f < -MAX_EXP) continue;
                sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                exr[c] = g * xy[c] * sigmoid * (1-sigmoid);
              }
            }
            //error of yr
            if (yr == xr) {
              for (c = 0; c < layer1_size; c++) eyr[c] = exr[c];
            }
            else {
              for (c = 0; c < layer1_size; c++) {
                if (sigmoid_reg)
                {
                  if (synr[c + lxr] < -MAX_EXP) continue; 
                  else if (synr[c + lxr] > MAX_EXP)
                  {
                    f = synr[c + lyr];
                    if (f > MAX_EXP || f < -MAX_EXP) continue;
                    sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                    eyr[c] = g * xy[c] * sigmoid * (1-sigmoid);
                  }
                  else
                  {
                    sigmoid2 = expTable[(int)((synr[c + lxr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                    f = synr[c + lxr];
                    if (f > MAX_EXP || f < -MAX_EXP) continue;
                    sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                    exr[c] = g * xy[c] * sigmoid * (1-sigmoid) * sigmoid2;
                  }
                }
                else
                {
                  if (synr[c + lxr] < 0) continue; 
                  f = synr[c + lyr];
                  if (f > MAX_EXP || f < -MAX_EXP) continue;
                  sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                  eyr[c] = g * xy[c] * sigmoid * (1-sigmoid);
                }
              }
            }
            //error of x and update y
            for (c = 0; c < layer1_size; c++) {
              if (sigmoid_reg) {
                if (synr[c + lxr] < -MAX_EXP || synr[c + lyr] < -MAX_EXP) continue;
                if (synr[c + lxr] > MAX_EXP) {
                  if (synr[c + lyr] > MAX_EXP) {
                    ex[c] = g * syn0[c + ly];
                    syn0[c + ly] += g * syn0[c + lx];
                  }
                  else {
                    sigmoid = expTable[(int)((synr[c + lyr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                    ex[c] = g * sigmoid * syn0[c + ly];
                    syn0[c + ly] += g * sigmoid * syn0[c + lx];
                  }
                }
                else if (synr[c + lyr] > MAX_EXP) {
                  sigmoid = expTable[(int)((synr[c + lxr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                  ex[c] = g * sigmoid * syn0[c + ly];
                  syn0[c + ly] += g * sigmoid * syn0[c + lx];
                }
                else {
                  sigmoid = expTable[(int)((synr[c + lxr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                  sigmoid2 = expTable[(int)((synr[c + lyr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                  ex[c] = g * sigmoid * sigmoid2 * syn0[c + ly];
                  syn0[c + ly] += g * sigmoid * sigmoid2 * syn0[c + lx];
                }
              }
              else
              {
                if (synr[c + lxr] >= 0 && synr[c + lyr] >= 0)
                {
                  ex[c] = g * syn0[c + ly];
                  syn0[c + ly] += g * syn0[c + lx];
                }
              }
            }
            //update x, xr, yr
            for (c = 0; c < layer1_size; c++)
            {
              syn0[c + lx] += ex[c];
              synr[c + lxr] += exr[c];
              synr[c + lyr] += eyr[c];
            }
          }

          //-------------------------------
          //Training of a data for graphlet
          //-------------------------------
          if (g_ratio == 0) continue;
          for (c = 0; c < layer1_size; c++) ex[c] = 0;
          for (c = 0; c < layer1_size; c++) eg[c] = 0;
          
          //Predicting
          f = 0;
          for (c = 0; c < layer1_size; c++) {
            if (sigmoid_reg) {
              if (syng[c + lg] < -MAX_EXP) continue;
              if (syng[c + lg] > MAX_EXP) {
                f += xy[c];
              } else {
                sigmoid = expTable[(int)((syng[c + lg] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                f += sigmoid * xy[c];
              }
            } else {
              if (syng[c + lg] >= 0)
              {
                f += xy[c];
              }
            }
          }
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          g = g * weight * g_ratio;
     
          //Updating
          //error of graphlet
          for (c = 0; c < layer1_size; c++) {
            f = syng[c + lg];
            if (f > MAX_EXP || f < -MAX_EXP) continue;
            sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            eg[c] = g * xy[c] * sigmoid * (1-sigmoid);
          }
          //error of x and update y
          for (c = 0; c < layer1_size; c++) {
            if (sigmoid_reg) {
              if (syng[c + lg] < -MAX_EXP) continue;
              if (syng[c + lg] > MAX_EXP) {
                ex[c] = g * syn0[c + ly];
                syn0[c + ly] += g * syn0[c + lx];
              } else {
                sigmoid = expTable[(int)((syng[c + lg] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                ex[c] = g * sigmoid * syn0[c + ly];
                syn0[c + ly] += g * sigmoid * syn0[c + lx];
              }
            }
            else
            {
              if (syng[c + lg] >= 0)
              {
                ex[c] = g * syn0[c + ly];
                syn0[c + ly] += g * syn0[c + lx];
              }
            }
          }
          //update x, graphlet 
          for (c = 0; c < layer1_size; c++)
          {
            syn0[c + lx] += ex[c];
            syng[c + lg] += eg[c];
          }
        }
      }
    }
  }
//printf("finish tid:%d\n", id);
  fclose(fi);
  free(ex);
  free(exr);
  free(eyr);
  free(eg);
  free(xy);
  pthread_exit(NULL);
}

int countlines()
{
  // count the number of lines in the file called filename                                    
  FILE *fp = fopen(train_file, "r");
  int ch=0;
  int lines=0;

  if (fp == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  } 

  while ((ch = fgetc(fp)) != EOF)
  {
    if (ch == '\n')
      lines++;
  }
  file_size = ftell(fp);
  fclose(fp);
  return lines;
}

void TrainModel() {
  long a, b;
  FILE *fo, *fo2, *fo3;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  if (pt == NULL) {
    fprintf(stderr, "cannot allocate memory for threads\n");
    exit(1);
  }
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;

  total_data_count = countlines();
  total_data_count *= iteration;
  printf("Data size: %lld\n", total_data_count);
  InitNet();
  InitUnigramTable();

  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

  fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
    exit(1);
  }
  printf("\nsave node vectors\n");
  fprintf(fo, "%lld %lld\n", node_count, layer1_size);
  for (a = 0; a < node_count; a++) {
    fprintf(fo, "%ld ", a);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);

  fo2 = fopen(role_output_file, "wb");
  if (fo2 == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", role_output_file);
    exit(1);
  }
  printf("save role vectors %s\n", role_output_file);
  fprintf(fo2, "%lld %lld\n", role_count, layer1_size);
  for (a = 0; a < role_count; a++) {
    fprintf(fo2, "%ld ", a);
    for (b = 0; b < layer1_size; b++) fprintf(fo2, "%lf ", synr[a * layer1_size + b]);
    fprintf(fo2, "\n");
  }
  fclose(fo2);
  fo3 = fopen(graphlet_output_file, "wb");
  if (fo3 == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", role_output_file);
    exit(1);
  }
  printf("save graphlet vectors %s\n", graphlet_output_file);
  fprintf(fo3, "%lld %lld\n", graphlet_count, layer1_size);
  for (a = 0; a < graphlet_count; a++) {
    fprintf(fo3, "%ld ", a);
    for (b = 0; b < layer1_size; b++) fprintf(fo3, "%lf ", syng[a * layer1_size + b]);
    fprintf(fo3, "\n");
  }
  fclose(fo3);
  free(table);
  free(pt);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("HIN representation learning\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of vectors; default is 100\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting node vectors\n");
    printf("\t-output_mp <file>\n");
    printf("\t\tUse <file> to save the resulting meta-path vectors\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max hop number of meta-paths between nodes; default is 3\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-sigmoid_reg <1/0>\n");
    printf("\t\tSet to use sigmoid function for regularization (default 0: use binary-step function)\n");
    printf("\nExamples:\n");
    printf("./hin2vec -train data.txt -output vec.txt -size 200 -window 5 -negative 5\n\n");
    return 0;
  }
  output_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-node_count", argc, argv)) > 0) node_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-role_count", argc, argv)) > 0) role_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-graphlet_count", argc, argv)) > 0) graphlet_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-role_ratio", argc, argv)) > 0) role_ratio = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-freq", argc, argv)) > 0) strcpy(freq_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output_role", argc, argv)) > 0) strcpy(role_output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output_graphlet", argc, argv)) > 0) strcpy(graphlet_output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sigmoid_reg", argc, argv)) > 0) sigmoid_reg = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iteration", argc, argv)) > 0) iteration = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-equal", argc, argv)) > 0) equal = atoi(argv[i + 1]);

  //Precompute exp table
  printf("node_count: %lld\n", node_count);
  printf("role_count: %lld\n", role_count);
  printf("graphlet_count: %lld\n", graphlet_count);
  g_ratio = 1.0 - role_ratio;
  printf("role_ratio: %f, graphlet_ratio: %f\n", role_ratio, g_ratio);
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (expTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  DestroyNet();
  free(expTable);
  return 0;
}
