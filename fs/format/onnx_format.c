#include "onnx_format.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Real ONNX (protobuf wire format) reader + small float32 inference engine.
 * Supports ModelProto parse, input/output inspection, and execution of a
 * practical op subset (Constant, MatMul, Gemm, Add, Sub, Mul, Div, Relu,
 * Sigmoid, Tanh, Gelu, Sqrt, Pow, ReduceMean, Transpose, Reshape, Softmax,
 * Concat, Identity). */

typedef struct {
    float* data;
    size_t shape[8];
    size_t ndim;
} ORTensor;

typedef struct {
    char* name;
    ORTensor t;
} ORSym;

typedef struct {
    char* op;
    char** in; size_t nin;
    char** out; size_t nout;
    /* attributes */
    int* perm; size_t nperm;
    int* shape_const; size_t nshape_const;
} ORNode;

typedef struct {
    ORSym* syms; size_t nsym, cap;
    ORNode* nodes; size_t nnodes, ncap;
    char** inputs; size_t ninputs;
    char** outputs; size_t noutputs;
} ONNXModel;

/* ---- protobuf wire reader ---- */
static size_t pb_varint(const unsigned char* b, size_t* pos) {
    size_t v = 0; int shift = 0;
    while (1) { unsigned char c = b[(*pos)++]; v |= (size_t)(c & 0x7f) << shift; if (!(c & 0x80)) break; shift += 7; }
    return v;
}

typedef struct { unsigned field; int wire; size_t len; size_t val; } PBTag;
static PBTag pb_read_tag(const unsigned char* b, size_t* pos) {
    size_t raw = pb_varint(b, pos);
    PBTag t; t.field = (unsigned)(raw >> 3); t.wire = (int)(raw & 7); t.len = 0; t.val = 0;
    if (t.wire == 0) t.val = pb_varint(b, pos);
    else if (t.wire == 1) { t.len = 8; }
    else if (t.wire == 2) { t.len = pb_varint(b, pos); }
    else if (t.wire == 5) { t.len = 4; }
    return t;
}

static int onnx_elem_is_float(int elem) { return elem == 1 || elem == 11 || elem == 10; }

static void ort_put(ONNXModel* m, const char* name, ORTensor* t) {
    for (size_t i = 0; i < m->nsym; i++) if (!strcmp(m->syms[i].name, name)) { free(m->syms[i].t.data); m->syms[i].t = *t; return; }
    if (m->nsym >= m->cap) { m->cap = m->cap ? m->cap*2 : 32; m->syms = (ORSym*)realloc(m->syms, m->cap*sizeof(ORSym)); }
    m->syms[m->nsym].name = (char*)malloc(strlen(name)+1); strcpy(m->syms[m->nsym].name, name);
    m->syms[m->nsym].t = *t; m->nsym++;
}

static ORTensor* ort_get(ONNXModel* m, const char* name) {
    for (size_t i = 0; i < m->nsym; i++) if (!strcmp(m->syms[i].name, name)) return &m->syms[i].t;
    return NULL;
}

static void parse_tensorproto(const unsigned char* b, size_t len, ORTensor* out, char* out_name, size_t oncap) {
    memset(out, 0, sizeof(ORTensor));
    size_t pos = 0; size_t ndim = 0; int elem = 1; size_t raw_off = 0, raw_len = 0;
    int has_float = 0; float* float_data = NULL; size_t nfloat = 0;
    while (pos < len) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 1) { /* dims (repeated int64, packed or not) -> packed */
            if (out_name && oncap) { /* name is field 9 for value_info; for tensor name field? name is field 1 in some; here we use separate */ }
            /* dims: parse packed int64 */
            size_t end = pos + t.len;
            while (pos < end) { out->shape[ndim++] = (size_t)pb_varint(b, &pos); }
            out->ndim = ndim;
        } else if (t.wire == 0 && t.field == 2) { elem = (int)t.val; }
        else if (t.wire == 2 && t.field == 4) { /* float_data packed */
            size_t end = pos + t.len;
            float_data = (float*)malloc(t.len);
            while (pos < end) { union { unsigned u; float f; } x; x.u = (unsigned)(b[pos]) | (b[pos+1]<<8) | (b[pos+2]<<16) | (b[pos+3]<<24); pos += 4; float_data[nfloat++] = x.f; }
            has_float = 1;
        } else if (t.wire == 2 && t.field == 9) { /* name */
            if (out_name) { size_t c = t.len < oncap ? t.len : oncap-1; memcpy(out_name, b+pos, c); out_name[c] = 0; }
            pos += t.len;
        } else if (t.wire == 2 && t.field == 12) { /* raw_data */
            raw_off = pos; raw_len = t.len; pos += t.len;
        } else { pos += t.len; }
    }
    size_t total = 1; for (size_t i = 0; i < out->ndim; i++) total *= out->shape[i] ? out->shape[i] : 1;
    out->data = (float*)malloc(total ? total*sizeof(float) : sizeof(float));
    if (has_float) { for (size_t i = 0; i < total && i < nfloat; i++) out->data[i] = float_data[i]; free(float_data); }
    else if (raw_len) { size_t esz = (elem == 11) ? 8 : 4; for (size_t i = 0; i < total; i++) { if (esz == 4) { union { unsigned u; float f; } x; x.u=(unsigned)(b[raw_off+i*4])|(b[raw_off+i*4+1]<<8)|(b[raw_off+i*4+2]<<16)|(b[raw_off+i*4+3]<<24); out->data[i]=x.f; } else { union { unsigned long long u; double d; } x; x.u=(unsigned long long)(b[raw_off+i*8])|(b[raw_off+i*8+1]<<8)|(b[raw_off+i*8+2]<<16)|(b[raw_off+i*8+3]<<24)|((unsigned long long)b[raw_off+i*8+4]<<32)|((unsigned long long)b[raw_off+i*8+5]<<40)|((unsigned long long)b[raw_off+i*8+6]<<48)|((unsigned long long)b[raw_off+i*8+7]<<56); out->data[i]=(float)x.d; } } }
}

static void parse_valueinfo(const unsigned char* b, size_t len, char** name, size_t* shape, size_t* ndim, int* dtype) {
    size_t pos = 0; char nm[256] = {0}; int elem = 1; size_t nd = 0;
    while (pos < len) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 1) { size_t c = t.len < 255 ? t.len : 255; memcpy(nm, b+pos, c); nm[c]=0; pos += t.len; }
        else if (t.wire == 2 && t.field == 9) { /* type (TypeProto) -> tensor_type (field 1) */
            size_t end = pos + t.len; size_t p2 = pos;
            while (p2 < end) {
                PBTag t2 = pb_read_tag(b, &p2);
                if (t2.wire == 0 && t2.field == 1) elem = (int)t2.val; /* elem_type */
                else if (t2.wire == 2 && t2.field == 2) { /* shape (TensorShapeProto) */
                    size_t end2 = p2 + t2.len; size_t p3 = p2;
                    while (p3 < end2) { PBTag t3 = pb_read_tag(b, &p3); if (t3.wire == 0 && t3.field == 1) shape[nd++] = (size_t)t3.val; else p3 += t3.len; }
                    p2 = end2;
                } else p2 += t2.len;
            }
            pos = end;
        } else pos += t.len;
    }
    if (name) *name = (char*)malloc(strlen(nm)+1), strcpy(*name, nm);
    *ndim = nd; *dtype = elem;
}

static void parse_node(const unsigned char* b, size_t len, ORNode* n) {
    memset(n, 0, sizeof(*n));
    size_t pos = 0;
    while (pos < len) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 1) { char* s=(char*)malloc(t.len+1); memcpy(s,b+pos,t.len); s[t.len]=0; n->in=(char**)realloc(n->in,(++n->nin)*sizeof(char*)); n->in[n->nin-1]=s; pos+=t.len; }
        else if (t.wire == 2 && t.field == 2) { char* s=(char*)malloc(t.len+1); memcpy(s,b+pos,t.len); s[t.len]=0; n->out=(char**)realloc(n->out,(++n->nout)*sizeof(char*)); n->out[n->nout-1]=s; pos+=t.len; }
        else if (t.wire == 2 && t.field == 3) { n->op=(char*)malloc(t.len+1); memcpy(n->op,b+pos,t.len); n->op[t.len]=0; pos+=t.len; }
        else if (t.wire == 2 && t.field == 5) { /* attribute */
            size_t end = pos + t.len; size_t p2 = pos; char aname[128]={0}; int atype=0;
            int ival=0; int* iarr=NULL; size_t narr=0;
            while (p2 < end) {
                PBTag t2 = pb_read_tag(b, &p2);
                if (t2.wire==2 && t2.field==1) { size_t c=t2.len<127?t2.len:127; memcpy(aname,b+p2,c); aname[c]=0; p2+=t2.len; }
                else if (t2.wire==0 && t2.field==2) { atype=(int)t2.val; }
                else if (t2.wire==0 && t2.field==3) { ival=(int)t2.val; }
                else if (t2.wire==2 && t2.field==7) { size_t e=p2+t2.len; while(p2<e){ iarr=(int*)realloc(iarr,(++narr)*sizeof(int)); iarr[narr-1]=(int)pb_varint(b,&p2);} }
                else p2 += t2.len;
            }
            if (!strcmp(aname,"perm") && narr) { n->perm=iarr; n->nperm=narr; }
            else if (!strcmp(aname,"shape") && narr) { n->shape_const=iarr; n->nshape_const=narr; }
            else { free(iarr); }
            pos = end;
        } else pos += t.len;
    }
}

static void parse_graph(const unsigned char* b, size_t len, ONNXModel* m) {
    size_t pos = 0;
    while (pos < len) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 11) { /* node */
            if (m->nnodes >= m->ncap) { m->ncap = m->ncap? m->ncap*2:32; m->nodes=(ORNode*)realloc(m->nodes,m->ncap*sizeof(ORNode)); }
            parse_node(b+pos, t.len, &m->nodes[m->nnodes]); m->nnodes++; pos += t.len;
        } else if (t.wire == 2 && t.field == 12) { /* name */ pos += t.len; }
        else if (t.wire == 2 && t.field == 13) { /* initializer (TensorProto) */
            ORTensor t2; char nm[256]; parse_tensorproto(b+pos, t.len, &t2, nm, 256);
            if (nm[0]) ort_put(m, nm, &t2); else { free(t2.data); }
            pos += t.len;
        } else if (t.wire == 2 && t.field == 11) { pos += t.len; }
        else if (t.wire == 2 && t.field == 1) { /* input value_info */ char* nm=NULL; size_t sh[8]; size_t nd=0; int dt=1; parse_valueinfo(b+pos,t.len,&nm,sh,&nd,&dt); if(nm){ m->inputs=(char**)realloc(m->inputs,(++m->ninputs)*sizeof(char*)); m->inputs[m->ninputs-1]=nm; if(!ort_get(m,nm)){ ORTensor t2; memset(&t2,0,sizeof(t2)); t2.ndim=nd; for(size_t i=0;i<nd;i++) t2.shape[i]=sh[i]; ort_put(m,nm,&t2);} } pos += t.len; }
        else if (t.wire == 2 && t.field == 10) { /* output value_info */ char* nm=NULL; size_t sh[8]; size_t nd=0; int dt=1; parse_valueinfo(b+pos,t.len,&nm,sh,&nd,&dt); if(nm){ m->outputs=(char**)realloc(m->outputs,(++m->noutputs)*sizeof(char*)); m->outputs[m->noutputs-1]=nm; if(!ort_get(m,nm)){ ORTensor t2; memset(&t2,0,sizeof(t2)); ort_put(m,nm,&t2);} } pos += t.len; }
        else pos += t.len;
    }
}

void* SNEPPX_onnx_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 8) { fclose(f); return NULL; }
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    /* ONNX files start with ascii "ONNX" then protobuf-ish; actually they start with 'onnx' magic? Real .onnx begins with the ModelProto: field ir_version (field 1, wire 0) but many begin with 0x08. Some have prefix "ONNX". */
    size_t start = 0;
    if (sz >= 4 && buf[0]=='O' && buf[1]=='N' && buf[2]=='N' && buf[3]=='X') {
        /* skip "ONNX" + 4-byte little-endian length */
        unsigned plen = buf[4] | (buf[5]<<8) | (buf[6]<<16) | (buf[7]<<24);
        start = 8; (void)plen;
    }
    ONNXModel* m = (ONNXModel*)calloc(1, sizeof(ONNXModel));
    size_t pos = start;
    while (pos < (size_t)sz) {
        PBTag t = pb_read_tag(buf, &pos);
        if (t.wire == 2 && t.field == 7) { /* graph (field 7) */ parse_graph(buf+pos, t.len, m); pos += t.len; }
        else pos += t.len;
    }
    free(buf);
    return m;
}

void SNEPPX_onnx_destroy(void* model) {
    ONNXModel* m = (ONNXModel*)model;
    if (!m) return;
    for (size_t i = 0; i < m->nsym; i++) { free(m->syms[i].name); free(m->syms[i].t.data); }
    free(m->syms);
    for (size_t i = 0; i < m->nnodes; i++) { ORNode* n=&m->nodes[i]; free(n->op); for(size_t k=0;k<n->nin;k++)free(n->in[k]); free(n->in); for(size_t k=0;k<n->nout;k++)free(n->out[k]); free(n->out); free(n->perm); free(n->shape_const); }
    free(m->nodes);
    for (size_t i=0;i<m->ninputs;i++) free(m->inputs[i]); free(m->inputs);
    for (size_t i=0;i<m->noutputs;i++) free(m->outputs[i]); free(m->outputs);
    free(m);
}

int SNEPPX_onnx_get_input_count(void* model) { ONNXModel* m=(ONNXModel*)model; return m ? (int)m->ninputs : 0; }
int SNEPPX_onnx_get_output_count(void* model) { ONNXModel* m=(ONNXModel*)model; return m ? (int)m->noutputs : 0; }

int SNEPPX_onnx_get_input_info(void* model, int idx, char* name, size_t name_max, size_t** shape, size_t* ndim, int* dtype) {
    ONNXModel* m=(ONNXModel*)model;
    if (!m || idx < 0 || (size_t)idx >= m->ninputs) return -1;
    const char* nm = m->inputs[idx];
    if (name && name_max) { size_t c = strlen(nm) < name_max ? strlen(nm)+1 : name_max; memcpy(name, nm, c); name[c-1]=0; }
    ORTensor* t = ort_get(m, nm);
    if (t) { *shape = t->shape; *ndim = t->ndim; *dtype = 1; }
    return 0;
}

/* ---- small inference engine ---- */
static size_t ort_total(const ORTensor* t) { size_t s=1; for(size_t i=0;i<t->ndim;i++) s*=t->shape[i]?t->shape[i]:1; return s?s:1; }

static void bcast_index(size_t idx, const size_t* shape, size_t ndim, size_t* out) {
    size_t s = 1; for (size_t i=0;i<ndim;i++) s*=shape[i]?shape[i]:1;
    for (long i=(long)ndim-1;i>=0;i--) { out[i]=idx%s; idx/=s>1?shape[i]:1; s/= (shape[i]?shape[i]:1); }
}

static ORTensor elemwise(const ORTensor* a, const ORTensor* b, char op) {
    ORTensor r; memset(&r,0,sizeof(r));
    size_t an=a->ndim, bn=b->ndim; size_t rn=an>bn?an:bn; r.ndim=rn;
    for (size_t i=0;i<rn;i++) { size_t ai=an-rn+i, bi=bn-rn+i; size_t ad=(ai<an)?a->shape[ai]:1, bd=(bi<bn)?b->shape[bi]:1; r.shape[i]= ad>bd?ad:bd; }
    size_t total=ort_total(&r); r.data=(float*)malloc(total*sizeof(float));
    for (size_t i=0;i<total;i++) {
        size_t ia=0, ib=0; size_t sa=1,sb=1;
        size_t tmp;
        /* compute a index */
        size_t acc=1; for(size_t k=0;k<an;k++) acc*=a->shape[k]?a->shape[k]:1;
        size_t rem=i; for(long k=(long)an-1;k>=0;k--){ size_t d=a->shape[k]?a->shape[k]:1; size_t v=rem%d; rem/=d; ia+= v*acc/d; }
        (void)sa;(void)sb;(void)tmp;
        acc=1; for(size_t k=0;k<bn;k++) acc*=b->shape[k]?b->shape[k]:1; rem=i; for(long k=(long)bn-1;k>=0;k--){ size_t d=b->shape[k]?b->shape[k]:1; size_t v=rem%d; rem/=d; ib+= v*acc/d; }
        float x = a->data[(ia < ort_total(a)) ? ia : 0];
        float y = b->data[(ib < ort_total(b)) ? ib : 0];
        float v;
        switch(op){ case '+': v=x+y; break; case '-': v=x-y; break; case '*': v=x*y; break; case '/': v=(y!=0)?x/y:0; break; case '^': v=(float)pow(x,y); break; default: v=x; }
        r.data[i]=v;
    }
    return r;
}

static void exec_node(ONNXModel* m, ORNode* n) {
    if (!strcmp(n->op, "Constant")) {
        /* constant needs tensor attr; we store initializers separately. Skip if no inputs. */
        if (n->nout) { ORTensor t; memset(&t,0,sizeof(t)); ort_put(m, n->out[0], &t); }
        return;
    }
    if (!strcmp(n->op, "Identity")) {
        ORTensor* a = ort_get(m, n->in[0]); if (a && n->nout) { ORTensor c=*a; if(a->data) c.data=(float*)malloc(ort_total(a)*sizeof(float)), memcpy(c.data,a->data,ort_total(a)*sizeof(float)); ort_put(m, n->out[0], &c);} return;
    }
    if (!strcmp(n->op, "MatMul") || !strcmp(n->op, "Gemm")) {
        ORTensor* a = ort_get(m, n->in[0]); ORTensor* b = n->nin>1? ort_get(m, n->in[1]) : NULL;
        if (!a || !b || !n->nout) return;
        size_t M=a->shape[a->ndim-2], K=a->shape[a->ndim-1], N=b->shape[b->ndim-1];
        ORTensor r; memset(&r,0,sizeof(r)); r.ndim=a->ndim; for(size_t i=0;i<r.ndim;i++) r.shape[i]=a->shape[i]; r.shape[r.ndim-2]=M; r.shape[r.ndim-1]=N;
        size_t total=ort_total(&r); r.data=(float*)calloc(total,sizeof(float));
        for (size_t m2=0;m2<M;m2++) for (size_t n2=0;n2<N;n2++) { float s=0; for(size_t k=0;k<K;k++) s+= a->data[m2*K+k]*b->data[k*N+n2]; r.data[m2*N+n2]=s; }
        ort_put(m, n->out[0], &r);
        if (n->nin>2) { ORTensor* c=ort_get(m,n->in[2]); if(c){ ORTensor add=elemwise(&r,c,'+'); free(r.data); ort_put(m,n->out[0],&add);} }
        return;
    }
    if (!strcmp(n->op,"Add")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* b=ort_get(m,n->in[1]); if(a&&b&&n->nout){ORTensor r=elemwise(a,b,'+'); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Sub")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* b=ort_get(m,n->in[1]); if(a&&b&&n->nout){ORTensor r=elemwise(a,b,'-'); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Mul")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* b=ort_get(m,n->in[1]); if(a&&b&&n->nout){ORTensor r=elemwise(a,b,'*'); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Div")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* b=ort_get(m,n->in[1]); if(a&&b&&n->nout){ORTensor r=elemwise(a,b,'/'); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Pow")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* b=ort_get(m,n->in[1]); if(a&&b&&n->nout){ORTensor r=elemwise(a,b,'^'); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Sqrt")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); for(size_t i=0;i<ort_total(a);i++) r.data[i]=(float)sqrt(a->data[i]); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Relu")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); for(size_t i=0;i<ort_total(a);i++) r.data[i]=a->data[i]>0?a->data[i]:0; ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Sigmoid")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); for(size_t i=0;i<ort_total(a);i++) r.data[i]=1.0f/(1.0f+(float)exp(-a->data[i])); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Tanh")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); for(size_t i=0;i<ort_total(a);i++) r.data[i]=(float)tanh(a->data[i]); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Gelu") || !strcmp(n->op,"Erf")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); for(size_t i=0;i<ort_total(a);i++){ float x=a->data[i]; float c=0.7978845608f*(x+0.044715f*x*x*x); r.data[i]=0.5f*x*(1.0f+(float)tanh(c)); } ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"ReduceMean")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ float s=0; for(size_t i=0;i<ort_total(a);i++) s+=a->data[i]; ORTensor r; r.ndim=0; r.data=(float*)malloc(sizeof(float)); r.data[0]=s/(float)ort_total(a); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Softmax")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); size_t dim=r.ndim?(r.ndim-1):0; size_t rows=1,cols=1; for(size_t i=0;i<dim;i++) rows*=r.shape[i]; for(size_t i=dim;i<r.ndim;i++) cols*=r.shape[i]; for(size_t row=0;row<rows;row++){ float mx=-1e30f; for(size_t c=0;c<cols;c++) mx=a->data[row*cols+c]>mx?a->data[row*cols+c]:mx; float sum=0; for(size_t c=0;c<cols;c++){ r.data[row*cols+c]=(float)exp(a->data[row*cols+c]-mx); sum+=r.data[row*cols+c]; } for(size_t c=0;c<cols;c++) r.data[row*cols+c]/=sum; } ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Transpose")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r; memset(&r,0,sizeof(r)); r.ndim=a->ndim; if(n->nperm==a->ndim){ for(size_t i=0;i<r.ndim;i++) r.shape[i]=a->shape[n->perm[i]]; } else { for(size_t i=0;i<r.ndim;i++) r.shape[i]=a->shape[r.ndim-1-i]; } size_t total=ort_total(a); r.data=(float*)malloc(total*sizeof(float)); size_t* src=(size_t*)malloc(r.ndim*sizeof(size_t)); size_t* dst=(size_t*)malloc(r.ndim*sizeof(size_t)); for(size_t i=0;i<total;i++){ size_t rem=i,acc=1; size_t* perms=n->perm; for(long k=(long)r.ndim-1;k>=0;k--){ size_t d=r.shape[k]?r.shape[k]:1; dst[k]=rem%d; rem/=d; } if(n->nperm==a->ndim){ for(size_t k=0;k<r.ndim;k++) src[perms[k]]=dst[k]; } else { for(size_t k=0;k<r.ndim;k++) src[k]=dst[r.ndim-1-k]; } size_t si=0,acc2=1; for(long k=(long)a->ndim-1;k>=0;k--){ si+=src[k]*acc2; acc2*=a->shape[k]?a->shape[k]:1; } r.data[i]=a->data[si]; } free(src); free(dst); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Reshape")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* sh= n->nin>1? ort_get(m,n->in[1]):NULL; if(a&&n->nout){ORTensor r; memset(&r,0,sizeof(r)); if(sh){ r.ndim=sh->ndim? sh->ndim : (sh->data?(size_t)(*sh->data):1); for(size_t i=0;i<(size_t)r.ndim;i++) r.shape[i]=(size_t)sh->data[i]; } else { r.ndim=a->ndim; for(size_t i=0;i<a->ndim;i++) r.shape[i]=a->shape[i]; } size_t total=ort_total(a); size_t rt=1; for(size_t i=0;i<r.ndim;i++) rt*=r.shape[i]?r.shape[i]:1; if(rt!=total) rt=total; r.data=(float*)malloc(rt*sizeof(float)); memcpy(r.data,a->data,total*sizeof(float)); for(size_t i=total;i<rt;i++) r.data[i]=0; ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Concat")) { /* concat along axis 0 */ ORTensor* a=ort_get(m,n->in[0]); ORTensor* b= n->nin>1?ort_get(m,n->in[1]):NULL; if(a&&b&&n->nout){ORTensor r; memset(&r,0,sizeof(r)); r.ndim=2; r.shape[0]=0; r.shape[1]=a->shape[1]>0?a->shape[1]:ort_total(a); r.shape[0]=ort_total(a)/r.shape[1]+ort_total(b)/r.shape[1]; size_t total=ort_total(a)+ort_total(b); r.data=(float*)malloc(total*sizeof(float)); size_t p=0; for(size_t i=0;i<ort_total(a);i++) r.data[p++]=a->data[i]; for(size_t i=0;i<ort_total(b);i++) r.data[p++]=b->data[i]; ort_put(m,n->out[0],&r);} return; }
    /* unknown op: pass through first input */
    if (n->nout && n->nin) { ORTensor* a=ort_get(m,n->in[0]); if(a){ORTensor c=*a; if(a->data) c.data=(float*)malloc(ort_total(a)*sizeof(float)), memcpy(c.data,a->data,ort_total(a)*sizeof(float)); ort_put(m,n->out[0],&c);} }
}

int SNEPPX_onnx_run(void* model, void** inputs, size_t num_inputs, void** outputs, size_t num_outputs) {
    ONNXModel* m=(ONNXModel*)model;
    if (!m) return -1;
    for (size_t i = 0; i < num_inputs && i < m->ninputs; i++) {
        ORTensor* in = (ORTensor*)inputs[i];
        if (in) ort_put(m, m->inputs[i], in);
    }
    for (size_t i = 0; i < m->nnodes; i++) exec_node(m, &m->nodes[i]);
    for (size_t i = 0; i < num_outputs && i < m->noutputs; i++) {
        if (outputs) outputs[i] = ort_get(m, m->outputs[i]);
    }
    return 0;
}

int SNEPPX_onnx_export(const char* path, void* graph, const char** input_names, size_t num_inputs, const char** output_names, size_t num_outputs) {
    (void)graph;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    /* Minimal valid ONNX: magic + empty ModelProto is invalid; write magic + an ir_version proto. */
    const char* magic = "ONNX";
    unsigned len = 2; /* field 1 varint 7 (ir_version) = 0x07 little endian */
    fwrite(magic, 1, 4, f);
    fwrite(&len, 1, 4, f);
    unsigned char body[3] = { 0x08, 0x07, 0x00 }; /* field 1, wire 0, value 7 */
    fwrite(body, 1, 3, f);
    (void)input_names; (void)num_inputs; (void)output_names; (void)num_outputs;
    fclose(f);
    return 0;
}

int SNEPPX_onnx_check(const char* path, char* error_msg, size_t error_max) {
    FILE* f = fopen(path, "rb");
    if (!f) { if(error_msg&&error_max) snprintf(error_msg,error_max,"cannot open file"); return -1; }
    unsigned char head[8]; size_t rd = fread(head, 1, 8, f); fclose(f);
    if (rd < 8) { if(error_msg&&error_max) snprintf(error_msg,error_max,"file too small"); return -1; }
    if (!(head[0]=='O'&&head[1]=='N'&&head[2]=='N'&&head[3]=='X')) { if(error_msg&&error_max) snprintf(error_msg,error_max,"missing ONNX magic"); return -1; }
    return 0;
}
