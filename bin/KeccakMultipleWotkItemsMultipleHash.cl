#define ROUNDS 24 
#define R64(a,b,c) (((a) << b) ^ ((a) >> c)) 

#define AS(j, i) A[i + j * BLOCK_SIZE]


typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

__constant unsigned long rc[5][ROUNDS] = {
    {0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
     0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
     0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
     0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
     0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
     0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
     0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
     0x8000000000008080, 0x0000000080000001, 0x8000000080008008},
    {0, 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
     0, 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
     0, 0 , 0 , 0 , 0 , 0 , 0 , 0 },
    {0, 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
     0, 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
     0, 0 , 0 , 0 , 0 , 0 , 0 , 0 },
    {0, 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
     0, 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
     0, 0 , 0 , 0 , 0 , 0 , 0 , 0 },
    {0, 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
     0, 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
     0, 0 , 0 , 0 , 0 , 0 , 0 , 0 }};

__constant unsigned int ro[25][2] = {
       /*y=0*/         /*y=1*/         /*y=2*/         /*y=3*/         /*y=4*/
/*x=0*/{ 0,64}, /*x=1*/{44,20}, /*x=2*/{43,21}, /*x=3*/{21,43}, /*x=4*/{14,50},
/*x=1*/{ 1,63}, /*x=2*/{ 6,58}, /*x=3*/{25,39}, /*x=4*/{ 8,56}, /*x=0*/{18,46},
/*x=2*/{62, 2}, /*x=3*/{55, 9}, /*x=4*/{39,25}, /*x=0*/{41,23}, /*x=1*/{ 2,62},
/*x=3*/{28,36}, /*x=4*/{20,44}, /*x=0*/{ 3,61}, /*x=1*/{45,19}, /*x=2*/{61, 3},
/*x=4*/{27,37}, /*x=0*/{36,28}, /*x=1*/{10,54}, /*x=2*/{15,49}, /*x=3*/{56, 8}};

__constant unsigned int a[25] = {
    0,  6, 12, 18, 24,
    1,  7, 13, 19, 20,
    2,  8, 14, 15, 21,
    3,  9, 10, 16, 22,
    4,  5, 11, 17, 23};

__constant unsigned int b[25] = {
    0,  1,  2,  3, 4,
    1,  2,  3,  4, 0,
    2,  3,  4,  0, 1,
    3,  4,  0,  1, 2,
    4,  0,  1,  2, 3};
    
__constant unsigned int c[25][3] = {
    { 0, 1, 2}, { 1, 2, 3}, { 2, 3, 4}, { 3, 4, 0}, { 4, 0, 1},
    { 5, 6, 7}, { 6, 7, 8}, { 7, 8, 9}, { 8, 9, 5}, { 9, 5, 6},
    {10,11,12}, {11,12,13}, {12,13,14}, {13,14,10}, {14,10,11},
    {15,16,17}, {16,17,18}, {17,18,19}, {18,19,15}, {19,15,16},
    {20,21,22}, {21,22,23}, {22,23,24}, {23,24,20}, {24,20,21}};
    
__constant unsigned int d[25] = {
          0,  1,  2,  3,  4,
         10, 11, 12, 13, 14,
         20, 21, 22, 23, 24,
          5,  6,  7,  8,  9,
         15, 16, 17, 18, 19};

unsigned long ROL64(unsigned long a, unsigned int offset)
{
    const int _offset = offset;
    return ((offset != 0) ? ((a << _offset) ^ (a >> (64-offset))) : a);
}


__kernel __attribute__ ((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
void KeccakMultipleWotkItemsMultipleHash(__global const unsigned long *d_data, 
            __global unsigned long *d_out) {

    __local unsigned long A[BLOCK_SIZE*BLOCK_SIZE];  
    __local unsigned long C[BLOCK_SIZE*BLOCK_SIZE]; 
    __local unsigned long D[BLOCK_SIZE*BLOCK_SIZE]; 

    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // index do primeiro sub-Bloco A
    int aBegin = WIDTH * BLOCK_SIZE * by;

    // Index do ultimo sub-Bloco A
    int aEnd   = aBegin + WIDTH - 1;

    // incremento para as iteracoes dos sub-Blocos A
    int aStep  = BLOCK_SIZE;    

    int t = tx + ty * BLOCK_SIZE; 
    int s = t%5;


    for (int a = aBegin; a <= aEnd; a += aStep) {
        
        AS(ty, tx) = d_data[a + WIDTH * ty + tx];
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
        //printf("%d ", tx);
        for(int i=0;i<ROUNDS;++i) { 

            C[t] = A[s]^A[s+5]^A[s+10]^A[s+15]^A[s+20];
            D[t] = C[b[20+s]] ^ R64(C[b[5+s]],1,63);
           
            if(t==0)
                C[0] = ROL64(A[0] ^ D[0], 0);
            else
                C[t] = R64(A[a[t]]^D[b[t]], ro[t][0], ro[t][1]);

            A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]); 
            A[t] = A[t]^rc[(t==0)?0:1][i];
           
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    d_out[get_global_id(1) * get_global_size(0) + get_global_id(0)] = A[t];

}
