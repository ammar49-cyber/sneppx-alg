#ifndef SNEPPX_HEARTBEAT_H
#define SNEPPX_HEARTBEAT_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
typedef SOCKET snepx_socket_t;
#define SNEPPX_INVALID_SOCKET INVALID_SOCKET
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
typedef int snepx_socket_t;
#define SNEPPX_INVALID_SOCKET (-1)
#endif

#define SNEPPX_HB_MAX_RANKS 1024
#define SNEPPX_HB_PORT_BASE 28700
#define SNEPPX_HB_MAGIC 0x534E5048

typedef enum {
    SNEPPX_HB_ALIVE = 0,
    SNEPPX_HB_SUSPECT = 1,
    SNEPPX_HB_DEAD = 2,
    SNEPPX_HB_JOIN = 3,
    SNEPPX_HB_LEAVE = 4
} SNEPPXHeartbeatStatus;

typedef struct {
    uint32_t magic;
    int32_t  rank;
    int32_t  world_size;
    SNEPPXHeartbeatStatus status;
    int64_t  timestamp_ns;
    uint32_t step;
    float    avg_loss;
    uint8_t  reserved[32];
} SNEPPXHeartbeatMessage;

typedef struct {
    int world_size;
    int rank;
    int heartbeat_interval_ms;
    int timeout_ms;

    snepx_socket_t socket;
    struct sockaddr_in* peer_addrs;
    int num_peers;

    SNEPPXHeartbeatStatus* peer_status;
    int64_t* last_heartbeat_ns;
    uint32_t* missed_count;

    int running;
    void* monitor_thread;

    int enable_tcp;
    char listen_addr[64];
    int listen_port;
} SNEPPXHeartbeat;

int  SNEPPX_heartbeat_init(SNEPPXHeartbeat** hb, int world_size, int rank,
                           int interval_ms, int timeout_ms);
int  SNEPPX_heartbeat_listen(SNEPPXHeartbeat* hb, const char* addr, int port);
int  SNEPPX_heartbeat_connect(SNEPPXHeartbeat* hb, int peer_rank,
                              const char* addr, int port);
int  SNEPPX_heartbeat_send(SNEPPXHeartbeat* hb, int peer_rank,
                           SNEPPXHeartbeatStatus status);
int  SNEPPX_heartbeat_recv(SNEPPXHeartbeat* hb, SNEPPXHeartbeatMessage* msg,
                           int timeout_ms);
int  SNEPPX_heartbeat_check_alive(SNEPPXHeartbeat* hb);
int  SNEPPX_heartbeat_get_alive_ranks(SNEPPXHeartbeat* hb, int* alive_ranks,
                                      int max_count);
void SNEPPX_heartbeat_destroy(SNEPPXHeartbeat* hb);

#endif
