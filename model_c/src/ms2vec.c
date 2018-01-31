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

char train_file[MAX_STRING], output_file[MAX_STRING], role_output_file[MAX_STRING], freq_file[MAX_STRING];
int binary = 0, debug_mode = 2, window = 3, num_threads = 1;
int sigmoid_reg = 0;
long long node_count = 0, layer1_size = 100, role_count = 0;
long long total_data_count = 0, data_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha;
real *syn0, *synr, *expTable;
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
}

void DestroyNet() {
}

void *TrainModelThread(void *id) {
  long long a, b;
  char item[MAX_STRING];
  long long data_count = 0, last_data_count = 0;
  real f, g;
  clock_t now;
  real sigmoid = 0;
  unsigned long long next_random = (long long)id;
  real *ex = (real *)calloc(layer1_size, sizeof(real));
  real *exr = (real *)calloc(layer1_size, sizeof(real));
  real *eyr = (real *)calloc(layer1_size, sizeof(real));
  long long lx, ly, lxr, lyr, c, label;
  char tokens[4][MAX_STRING];
  char *token;
  int token_length = 0;
  int x_size = 0, xr_size = 0;
  int xs[window];
  int xrs[window];
  int x, xr, y, yr, pos_y;
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
      if (token_length != 4) continue;
     
      //Parsing
      pos_y = atoi(tokens[0]);
      yr = atoi(tokens[1]);
//    printf("y:%d yr:%d ", y, yr);
      token = strtok_r(tokens[2], ",", &pSave);
      x_size = 0;
      while (token != NULL)
      {
        xs[x_size] = atoi(token);
//      printf("x%d:%d ", x_size, xs[x_size]);
        x_size++;
        token = strtok_r(NULL, ",", &pSave);
      }
      token = strtok_r(tokens[3], ",", &pSave);
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
     
      //Learning
      if (equal) weight = 1.0/x_size;
      for (a=0; a<x_size; a++)
      {
        x = xs[a];
        xr = xrs[a];
     
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
     
          //Training of a data
          lx = x * layer1_size;
          ly = y * layer1_size;
          lxr = xr * layer1_size;
          for (c = 0; c < layer1_size; c++) ex[c] = 0;
          for (c = 0; c < layer1_size; c++) exr[c] = 0;
          for (c = 0; c < layer1_size; c++) eyr[c] = 0;
          
          //Predicting
          f = 0;
          for (c = 0; c < layer1_size; c++) {
            //TODO sigmoid_reg
//          if (sigmoid_reg) {
//            if (synmp[c + lr] > MAX_EXP) f += syn0[c + lx] * syn0[c + ly];
//            else if (synmp[c + lr] < -MAX_EXP) continue;
//            else f += syn0[c + lx] * syn0[c + ly] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
//          } else {
              if (synr[c + lxr] >= 0 && synr[c + lyr] >= 0)
              {
                f += syn0[c + lx] * syn0[c + ly];
              }
//          }
          }
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          g *= weight;
     
          //Updating
          //error of x
          for (c = 0; c < layer1_size; c++) {
            //TODO
//          if (sigmoid_reg) {
//            if (synmp[c + lr] > MAX_EXP) ex[c] = g * syn0[c + ly];
//            else if (synmp[c + lr] < -MAX_EXP) continue;
//            else ex[c] = g * syn0[c + ly] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
//          } else {
              if (synr[c + lxr] >= 0 && synr[c + lyr] >= 0)
              {
                ex[c] = g * syn0[c + ly];
              }
//          }
          }
          //error of xr
          //TODO sigmoid reg
          for (c = 0; c < layer1_size; c++) {
            if (synr[c + lyr] < 0) continue; 
            f = synr[c + lxr];
            if (f > MAX_EXP || f < -MAX_EXP) continue;
            sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            exr[c] = g * syn0[c + lx] * syn0[c + ly] * sigmoid * (1-sigmoid);
          }
          //error of yr
          //TODO sigmoid reg
          if (yr == xr) {
            for (c = 0; c < layer1_size; c++) eyr[c] = exr[c];
          }
          else {
            for (c = 0; c < layer1_size; c++) {
              if (synr[c + lxr] < 0) continue; 
              f = synr[c + lyr];
              if (f > MAX_EXP || f < -MAX_EXP) continue;
              sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
              eyr[c] = g * syn0[c + lx] * syn0[c + ly] * sigmoid * (1-sigmoid);
            }
          }
          //update y
          for (c = 0; c < layer1_size; c++) {
//          if (sigmoid_reg) {
//            if (synmp[c + lr] > MAX_EXP) syn0[c + ly] += g * syn0[c + lx];
//            else if (synmp[c + lr] < -MAX_EXP) continue;
//            else syn0[c + ly] += g * syn0[c + lx] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
//          } else {
              if (synr[c + lxr] >= 0 && synr[c + lyr] >= 0)
              {
                syn0[c + ly] += g * syn0[c + lx];
              }
//          }
          }
          //update x, xr, yr
          for (c = 0; c < layer1_size; c++)
          {
            syn0[c + lx] += ex[c];
            synr[c + lxr] += exr[c];
            synr[c + lyr] += eyr[c];
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
  FILE *fo;
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
//  fo_mp = fopen(mp_output_file, "wb");
//  if (fo == NULL) {
//    fprintf(stderr, "Cannot open %s: permission denied\n", mp_output_file);
//    exit(1);
//  }
//  printf("save mp vectors\n");
//  fprintf(fo_mp, "%lld %lld\n", mp_vocab_size, layer1_size);
//  for (a = 0; a < mp_vocab_size; a++) {
//    if (mp_vocab[a].mp != NULL) {
//      fprintf(fo_mp, "%s ", mp_vocab[a].mp);
//    }
//    for (b = 0; b < layer1_size; b++) fprintf(fo_mp, "%lf ", synr[a * layer1_size + b]);
//    fprintf(fo_mp, "\n");
//  }
  fclose(fo);
//  fclose(fo_mp);
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
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-freq", argc, argv)) > 0) strcpy(freq_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output_role", argc, argv)) > 0) strcpy(role_output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sigmoid_reg", argc, argv)) > 0) sigmoid_reg = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iteration", argc, argv)) > 0) iteration = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-equal", argc, argv)) > 0) equal = atoi(argv[i + 1]);

  //Precompute exp table
  printf("node_count: %lld\n", node_count);
  printf("role_count: %lld\n", role_count);
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
