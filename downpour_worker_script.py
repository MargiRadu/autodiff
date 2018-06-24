from optimization.downpour.downpour_worker import DownpourWorkerServer


def main():
    port_nos = [16001, 16002, 16003, 16004, 16005]
    worker = DownpourWorkerServer(port_nos[1])


if __name__ == '__main__':
    main()