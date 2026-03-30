/**
 * WebSocket Transport — broadcasts GSD events as JSON over WebSocket.
 *
 * Implements TransportHandler. Starts a WebSocketServer on a given port
 * and JSON-serializes each event to all connected clients.
 */

import { WebSocketServer, WebSocket } from 'ws';
import fs from 'fs';
import fsPromises from 'fs/promises';
import os from 'os';
import path from 'path';
import type { GSDEvent, TransportHandler } from './types.js';

export interface WSTransportOptions {
  port: number;
  projectDir: string;
  workstream: string;
  totalPhases?: number;
}

export class WSTransport implements TransportHandler {
  private readonly port: number;
  private readonly options: WSTransportOptions;
  private server: WebSocketServer | null = null;
  private closing = false;
  private pidFile: string | null = null;

  constructor(options: WSTransportOptions) {
    this.port = options.port;
    this.options = options;
  }

  /**
   * Start the WebSocket server on the configured port.
   * Resolves once the server is listening.
   */
  async start(): Promise<void> {
    if (this.closing) return;

    await new Promise<void>((resolve, reject) => {
      try {
        this.server = new WebSocketServer({ port: this.port });
        this.server.on('listening', () => resolve());
        this.server.on('error', (err) => reject(err));
      } catch (err) {
        reject(err);
      }
    });

    // Write PID file for auto-discovery
    const instancesDir = path.join(os.homedir(), '.gsd', 'instances');
    await fsPromises.mkdir(instancesDir, { recursive: true });
    const pidFile = path.join(instancesDir, `${process.pid}.json`);
    await fsPromises.writeFile(pidFile, JSON.stringify({
      port: this.port,
      pid: process.pid,
      projectDir: this.options.projectDir,
      workstream: this.options.workstream,
      startedAt: new Date().toISOString(),
      ...(this.options.totalPhases !== undefined && { totalPhases: this.options.totalPhases }),
    }));
    this.pidFile = pidFile;
  }

  /**
   * Broadcast a GSD event as JSON to all connected clients.
   * Never throws — wraps each client.send in try/catch.
   */
  onEvent(event: GSDEvent): void {
    try {
      if (!this.server) return;

      const payload = JSON.stringify(event);

      for (const client of this.server.clients) {
        if (client.readyState === WebSocket.OPEN) {
          try {
            client.send(payload);
          } catch {
            // Ignore individual client send errors
          }
        }
      }
    } catch {
      // TransportHandler contract: onEvent must never throw
    }
  }

  /**
   * Close all client connections and shut down the server.
   * Safe to call before start() — sets a closing flag.
   */
  close(): void {
    this.closing = true;

    // Remove PID file synchronously (close() must remain void per TransportHandler)
    if (this.pidFile) {
      try {
        fs.unlinkSync(this.pidFile);
      } catch {
        // already removed or never written — ignore
      }
      this.pidFile = null;
    }

    if (!this.server) return;

    // Terminate all clients
    for (const client of this.server.clients) {
      try {
        client.terminate();
      } catch {
        // Ignore client close errors
      }
    }

    // Close the server
    try {
      this.server.close();
    } catch {
      // Ignore server close errors
    }

    this.server = null;
  }
}
