#include "http_server.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  typedef SOCKET sock_t;
  #define SNEPPX_INVALID_SOCKET INVALID_SOCKET
  #define snepx_close closesocket
  #define snepx_sleep(ms) Sleep(ms)
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #include <pthread.h>
  #include <dirent.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  typedef int sock_t;
  #define SNEPPX_INVALID_SOCKET (-1)
  #define snepx_close close
  #define snepx_sleep(ms) usleep((ms) * 1000)
#endif

#define MAX_ROUTES 128
#define MAX_MIDDLEWARE 32
#define MAX_THREADS 16
#define BUF_SIZE 65536
#define MAX_HEADERS 64

/* ---- Request / Response ---- */

struct SNEPPX_HttpRequest {
    char method[16];
    char path[1024];
    char query[1024];
    char headers[MAX_HEADERS][256];
    int num_headers;
    char body[BUF_SIZE];
    size_t body_len;
    void* userdata;
    sock_t client_fd;
    SNEPPX_HttpServer* server;
};

struct SNEPPX_HttpResponse {
    int status_code;
    char headers[MAX_HEADERS][256];
    int num_headers;
    char body[BUF_SIZE];
    size_t body_len;
};

/* ---- Route / Middleware ---- */

typedef struct {
    char method[16];
    char path_pattern[256];
    SNEPPX_http_handler_fn handler;
    void* userdata;
} RouteEntry;

typedef struct {
    SNEPPX_http_middleware_fn fn;
    void* userdata;
} MiddlewareEntry;

struct SNEPPX_HttpServer {
    int port;
    int num_threads;
    volatile int running;
    sock_t listen_fd;
    RouteEntry routes[MAX_ROUTES];
    int num_routes;
    MiddlewareEntry middleware[MAX_MIDDLEWARE];
    int num_middleware;
    char static_dirs[MAX_ROUTES][2][256];
    int num_static_dirs;
    void* thread_handles[MAX_THREADS];
    int num_threads_started;
};

/* ---- Utility ---- */

static int str_starts_with(const char* s, const char* prefix) {
    while (*prefix) { if (*s++ != *prefix++) return 0; }
    return 1;
}

static void split_query(const char* path, char* out_path, char* out_query) {
    const char* q = strchr(path, '?');
    if (q) {
        size_t plen = (size_t)(q - path);
        if (plen > 1023) plen = 1023;
        memcpy(out_path, path, plen);
        out_path[plen] = '\0';
        strncpy(out_query, q + 1, 1023);
        out_query[1023] = '\0';
    } else {
        strncpy(out_path, path, 1023);
        out_path[1023] = '\0';
        out_query[0] = '\0';
    }
}

static int match_route(const char* pattern, const char* path) {
    /* Simple prefix match for now (no wildcards) */
    return strcmp(pattern, path) == 0;
}

static const char* find_header(SNEPPX_HttpRequest* req, const char* name) {
    for (int i = 0; i < req->num_headers; i++) {
        if (str_starts_with(req->headers[i], name) && req->headers[i][strlen(name)] == ':')
            return req->headers[i] + strlen(name) + 2;
    }
    return NULL;
}

static void add_header(SNEPPX_HttpResponse* resp, const char* name, const char* value) {
    if (resp->num_headers >= MAX_HEADERS) return;
    snprintf(resp->headers[resp->num_headers++], 256, "%s: %s", name, value);
}

/* ---- Parse HTTP request ---- */

static int parse_http_request(const char* raw, size_t raw_len, SNEPPX_HttpRequest* req) {
    memset(req, 0, sizeof(*req));
    /* Parse request line: METHOD PATH HTTP/1.x */
    char line[8192];
    size_t line_end = 0;
    while (line_end < raw_len && raw[line_end] != '\r' && raw[line_end] != '\n') line_end++;
    if (line_end >= sizeof(line)) return -1;
    memcpy(line, raw, line_end);
    line[line_end] = '\0';
    sscanf(line, "%15s %1023s", req->method, req->path);
    /* Split path and query */
    char full_path[1024];
    strncpy(full_path, req->path, 1023);
    full_path[1023] = '\0';
    split_query(full_path, req->path, req->query);
    /* Parse headers */
    const char* hdr_start = raw + line_end;
    while (*hdr_start == '\r' || *hdr_start == '\n') hdr_start++;
    const char* h = hdr_start;
    while (req->num_headers < MAX_HEADERS) {
        const char* nl = strstr(h, "\r\n");
        if (!nl || nl == h) break;
        size_t hlen = (size_t)(nl - h);
        if (hlen > 255) hlen = 255;
        memcpy(req->headers[req->num_headers], h, hlen);
        req->headers[req->num_headers][hlen] = '\0';
        req->num_headers++;
        h = nl + 2;
    }
    /* Body is after \r\n\r\n */
    const char* body_start = strstr(raw, "\r\n\r\n");
    if (body_start) {
        body_start += 4;
        size_t body_avail = raw_len - (size_t)(body_start - raw);
        if (body_avail > BUF_SIZE - 1) body_avail = BUF_SIZE - 1;
        memcpy(req->body, body_start, body_avail);
        req->body[body_avail] = '\0';
        req->body_len = body_avail;
    }
    return 0;
}

/* ---- Build HTTP response ---- */

static int build_response(SNEPPX_HttpResponse* resp, char* out, size_t out_max) {
    const char* status_text = "OK";
    int sc = resp->status_code;
    if (sc == 404) status_text = "Not Found";
    else if (sc == 401) status_text = "Unauthorized";
    else if (sc == 403) status_text = "Forbidden";
    else if (sc == 500) status_text = "Internal Server Error";
    else if (sc == 429) status_text = "Too Many Requests";
    else if (sc == 400) status_text = "Bad Request";

    /* Default content-type if not set */
    int has_content_type = 0;
    for (int i = 0; i < resp->num_headers; i++) {
        if (str_starts_with(resp->headers[i], "Content-Type")) { has_content_type = 1; break; }
    }

    size_t pos = 0;
    int n = snprintf(out, out_max, "HTTP/1.1 %d %s\r\n", sc, status_text);
    if (n > 0) pos += (size_t)n;
    for (int i = 0; i < resp->num_headers; i++) {
        n = snprintf(out + pos, out_max - pos, "%s\r\n", resp->headers[i]);
        if (n > 0) pos += (size_t)n;
    }
    if (!has_content_type) {
        n = snprintf(out + pos, out_max - pos, "Content-Type: text/plain\r\n");
        if (n > 0) pos += (size_t)n;
    }
    n = snprintf(out + pos, out_max - pos, "Content-Length: %u\r\n\r\n", (unsigned)resp->body_len);
    if (n > 0) pos += (size_t)n;
    if (resp->body_len > 0 && pos + resp->body_len < out_max) {
        memcpy(out + pos, resp->body, resp->body_len);
        pos += resp->body_len;
    }
    out[pos] = '\0';
    return (int)pos;
}

/* ---- Handle single connection ---- */

static void handle_connection(sock_t client_fd, SNEPPX_HttpServer* srv) {
    char raw[BUF_SIZE];
    int n = recv(client_fd, raw, (int)sizeof(raw) - 1, 0);
    if (n <= 0) { snepx_close(client_fd); return; }
    raw[n] = '\0';

    SNEPPX_HttpRequest req;
    SNEPPX_HttpResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.status_code = 200;

    if (parse_http_request(raw, (size_t)n, &req) != 0) {
        resp.status_code = 400;
        const char* err = "Bad Request";
        SNEPPX_http_response_set_body_str(&resp, err);
        char out[BUF_SIZE];
        int len = build_response(&resp, out, sizeof(out));
        send(client_fd, out, len, 0);
        snepx_close(client_fd);
        return;
    }
    req.client_fd = client_fd;
    req.server = srv;

    /* Run middleware chain */
    int abort = 0;
    for (int i = 0; i < srv->num_middleware; i++) {
        resp.status_code = 200;
        resp.body_len = 0;
        resp.num_headers = 0;
        if (srv->middleware[i].fn(&req, &resp, srv->middleware[i].userdata) != 0) {
            abort = 1;
            break;
        }
    }

    /* Find and run handler (if not aborted) */
    if (!abort) {
        int handled = 0;
        /* Check static dirs */
        for (int i = 0; i < srv->num_static_dirs; i++) {
            if (str_starts_with(req.path, srv->static_dirs[i][0])) {
                const char* rel = req.path + strlen(srv->static_dirs[i][0]);
                if (*rel == '/') rel++;
                char filepath[512];
                snprintf(filepath, sizeof(filepath), "%s/%s", srv->static_dirs[i][1], rel);
                FILE* f = fopen(filepath, "rb");
                if (f) {
                    fseek(f, 0, SEEK_END);
                    long flen = ftell(f);
                    fseek(f, 0, SEEK_SET);
                    if (flen > 0 && (size_t)flen < BUF_SIZE) {
                        resp.body_len = (size_t)fread(resp.body, 1, (size_t)flen, f);
                        add_header(&resp, "Content-Type", "application/octet-stream");
                        const char* ext = strrchr(filepath, '.');
                        if (ext) {
                            if (strcmp(ext, ".html") == 0) add_header(&resp, "Content-Type", "text/html");
                            else if (strcmp(ext, ".css") == 0) add_header(&resp, "Content-Type", "text/css");
                            else if (strcmp(ext, ".js") == 0) add_header(&resp, "Content-Type", "application/javascript");
                            else if (strcmp(ext, ".json") == 0) add_header(&resp, "Content-Type", "application/json");
                            else if (strcmp(ext, ".png") == 0) add_header(&resp, "Content-Type", "image/png");
                            else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0) add_header(&resp, "Content-Type", "image/jpeg");
                        }
                    }
                    fclose(f);
                    handled = 1;
                    break;
                }
            }
        }
        /* Check registered routes */
        if (!handled) {
            for (int i = 0; i < srv->num_routes; i++) {
                if (strcmp(req.method, srv->routes[i].method) == 0 &&
                    match_route(srv->routes[i].path_pattern, req.path)) {
                    resp.status_code = 200;
                    resp.body_len = 0;
                    resp.num_headers = 0;
                    if (srv->routes[i].handler(&req, &resp, srv->routes[i].userdata) != 0) {
                        if (resp.status_code == 200) resp.status_code = 500;
                    }
                    handled = 1;
                    break;
                }
            }
        }
        if (!handled) {
            resp.status_code = 404;
            const char* msg = "Not Found";
            SNEPPX_http_response_set_body_str(&resp, msg);
        }
    }

    char out[BUF_SIZE];
    int len = build_response(&resp, out, sizeof(out));
    send(client_fd, out, len, 0);
    snepx_close(client_fd);
}

/* ---- Thread worker ---- */

#ifdef _WIN32
DWORD WINAPI worker_thread(LPVOID arg) {
    sock_t* pfd = (sock_t*)arg;
    sock_t client_fd = *pfd;
    SNEPPX_HttpServer* srv = (SNEPPX_HttpServer*)(((void**)arg)[1]);
    handle_connection(client_fd, srv);
    free(arg);
    return 0;
}
#else
static void* worker_thread(void* arg) {
    sock_t* pfd = (sock_t*)arg;
    sock_t client_fd = *pfd;
    SNEPPX_HttpServer* srv = (SNEPPX_HttpServer*)(((void**)arg)[1]);
    handle_connection(client_fd, srv);
    free(arg);
    return NULL;
}
#endif

/* ---- API Implementation ---- */

SNEPPX_HttpServer* SNEPPX_http_server_create(int port, int num_threads) {
    SNEPPX_HttpServer* srv = (SNEPPX_HttpServer*)calloc(1, sizeof(*srv));
    if (!srv) return NULL;
    srv->port = port;
    srv->num_threads = num_threads < 1 ? 1 : (num_threads > MAX_THREADS ? MAX_THREADS : num_threads);
    srv->running = 0;
    srv->listen_fd = SNEPPX_INVALID_SOCKET;
    return srv;
}

int SNEPPX_http_server_start(SNEPPX_HttpServer* srv) {
    if (!srv) return -1;
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return -1;
#endif
    srv->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv->listen_fd == SNEPPX_INVALID_SOCKET) return -1;
    int opt = 1;
#ifdef _WIN32
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((unsigned short)srv->port);
    if (bind(srv->listen_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        snepx_close(srv->listen_fd); srv->listen_fd = SNEPPX_INVALID_SOCKET; return -1;
    }
    if (listen(srv->listen_fd, 128) != 0) {
        snepx_close(srv->listen_fd); srv->listen_fd = SNEPPX_INVALID_SOCKET; return -1;
    }
    srv->running = 1;

    /* Accept loop — spawn threads per connection */
    while (srv->running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        sock_t client_fd = accept(srv->listen_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (client_fd == SNEPPX_INVALID_SOCKET) {
            if (!srv->running) break;
            snepx_sleep(10);
            continue;
        }
        /* Allocate args for thread */
        void** args = (void**)malloc(2 * sizeof(void*));
        if (!args) { snepx_close(client_fd); continue; }
        sock_t* pfd = (sock_t*)malloc(sizeof(sock_t));
        if (!pfd) { free(args); snepx_close(client_fd); continue; }
        *pfd = client_fd;
        args[0] = (void*)pfd;
        args[1] = (void*)srv;
#ifdef _WIN32
        HANDLE th = CreateThread(NULL, 0, worker_thread, args, 0, NULL);
        if (th) { CloseHandle(th); srv->num_threads_started++; }
        else { free(pfd); free(args); snepx_close(client_fd); }
#else
        pthread_t tid;
        if (pthread_create(&tid, NULL, worker_thread, args) == 0) {
            pthread_detach(tid);
            srv->num_threads_started++;
        } else {
            free(pfd); free(args); snepx_close(client_fd);
        }
#endif
    }
    return 0;
}

void SNEPPX_http_server_stop(SNEPPX_HttpServer* srv) {
    if (!srv) return;
    srv->running = 0;
    if (srv->listen_fd != SNEPPX_INVALID_SOCKET) {
        snepx_close(srv->listen_fd);
        srv->listen_fd = SNEPPX_INVALID_SOCKET;
    }
}

void SNEPPX_http_server_destroy(SNEPPX_HttpServer* srv) {
    if (!srv) return;
    SNEPPX_http_server_stop(srv);
#ifdef _WIN32
    WSACleanup();
#endif
    free(srv);
}

int SNEPPX_http_server_add_route(SNEPPX_HttpServer* srv, const char* method, const char* path,
                                  SNEPPX_http_handler_fn handler, void* userdata) {
    if (!srv || srv->num_routes >= MAX_ROUTES) return -1;
    int i = srv->num_routes++;
    strncpy(srv->routes[i].method, method, 15);
    srv->routes[i].method[15] = '\0';
    strncpy(srv->routes[i].path_pattern, path, 255);
    srv->routes[i].path_pattern[255] = '\0';
    srv->routes[i].handler = handler;
    srv->routes[i].userdata = userdata;
    return 0;
}

int SNEPPX_http_server_add_static_dir(SNEPPX_HttpServer* srv, const char* url_prefix, const char* dir_path) {
    if (!srv || srv->num_static_dirs >= MAX_ROUTES) return -1;
    int i = srv->num_static_dirs++;
    strncpy(srv->static_dirs[i][0], url_prefix, 255);
    srv->static_dirs[i][0][255] = '\0';
    strncpy(srv->static_dirs[i][1], dir_path, 255);
    srv->static_dirs[i][1][255] = '\0';
    return 0;
}

int SNEPPX_http_server_add_middleware(SNEPPX_HttpServer* srv, SNEPPX_http_middleware_fn mw, void* userdata) {
    if (!srv || srv->num_middleware >= MAX_MIDDLEWARE) return -1;
    int i = srv->num_middleware++;
    srv->middleware[i].fn = mw;
    srv->middleware[i].userdata = userdata;
    return 0;
}

/* ---- Request accessors ---- */

const char* SNEPPX_http_request_method(SNEPPX_HttpRequest* req) { return req->method; }
const char* SNEPPX_http_request_path(SNEPPX_HttpRequest* req) { return req->path; }
const char* SNEPPX_http_request_header(SNEPPX_HttpRequest* req, const char* name) { return find_header(req, name); }
const char* SNEPPX_http_request_body(SNEPPX_HttpRequest* req, size_t* len) { if (len) *len = req->body_len; return req->body; }
const char* SNEPPX_http_request_query(SNEPPX_HttpRequest* req, const char* key) {
    if (!req->query[0] || !key) return NULL;
    const char* q = req->query;
    while (*q) {
        while (*q == '&') q++;
        const char* eq = strchr(q, '=');
        const char* amp = strchr(q, '&');
        size_t klen = strlen(key);
        size_t qlen = eq ? (size_t)(eq - q) : (amp ? (size_t)(amp - q) : strlen(q));
        if (klen == qlen && memcmp(q, key, klen) == 0) {
            return eq ? eq + 1 : "";
        }
        if (amp) q = amp + 1; else break;
    }
    return NULL;
}
void* SNEPPX_http_request_userdata(SNEPPX_HttpRequest* req) { return req->userdata; }

/* ---- Response setters ---- */

void SNEPPX_http_response_set_status(SNEPPX_HttpResponse* resp, int status_code) { resp->status_code = status_code; }
void SNEPPX_http_response_set_header(SNEPPX_HttpResponse* resp, const char* name, const char* value) {
    add_header(resp, name, value);
}
void SNEPPX_http_response_set_body(SNEPPX_HttpResponse* resp, const char* data, size_t len) {
    if (len > BUF_SIZE - 1) len = BUF_SIZE - 1;
    memcpy(resp->body, data, len);
    resp->body[len] = '\0';
    resp->body_len = len;
}
void SNEPPX_http_response_set_body_str(SNEPPX_HttpResponse* resp, const char* str) {
    SNEPPX_http_response_set_body(resp, str, strlen(str));
}
void SNEPPX_http_response_set_json(SNEPPX_HttpResponse* resp, const char* json_str) {
    add_header(resp, "Content-Type", "application/json");
    SNEPPX_http_response_set_body_str(resp, json_str);
}
