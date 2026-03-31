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

export interface StateSnapshot {
  type: 'state_snapshot';
  timestamp: string;
  sessionId: string;
  currentPhase: string | null;
  currentPhaseName: string | null;
  currentStep: string | null;
  completedSteps: Array<{ step: string; durationMs: number; costUsd: number }>;
  completedPhases: Array<{ phaseNumber: string; phaseName: string; success: boolean }>;
  sessionCostUsd: number;
  cumulativeCostUsd: number;
  model: string | null;
  status: 'running' | 'complete' | 'error';
}

export class WSTransport implements TransportHandler {
  private readonly port: number;
  private readonly options: WSTransportOptions;
  private server: WebSocketServer | null = null;
  private closing = false;
  private pidFile: string | null = null;
  private stateSnapshot: StateSnapshot = {
    type: 'state_snapshot',
    timestamp: new Date().toISOString(),
    sessionId: '',
    currentPhase: null,
    currentPhaseName: null,
    currentStep: null,
    completedSteps: [],
    completedPhases: [],
    sessionCostUsd: 0,
    cumulativeCostUsd: 0,
    model: null,
    status: 'running',
  };

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

    // Send state snapshot to newly connected clients
    this.server!.on('connection', (client: WebSocket) => {
      try {
        if (client.readyState === WebSocket.OPEN) {
          client.send(JSON.stringify({
            ...this.stateSnapshot,
            timestamp: new Date().toISOString(),
          }));
        }
      } catch {
        // Ignore send errors on new connections
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

      // Update state snapshot based on event type
      this.updateSnapshot(event);

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
   * Update the accumulated state snapshot from an incoming event.
   */
  private updateSnapshot(event: GSDEvent): void {
    const snap = this.stateSnapshot;
    snap.sessionId = event.sessionId;
    snap.timestamp = event.timestamp;

    switch (event.type) {
      case 'session_init': {
        const ev = event as unknown as Record<string, unknown>;
        snap.model = (ev.model as string) ?? null;
        snap.status = 'running';
        break;
      }
      case 'phase_start': {
        const ev = event as unknown as Record<string, unknown>;
        snap.currentPhase = (ev.phaseNumber as string) ?? null;
        snap.currentPhaseName = (ev.phaseName as string) ?? null;
        snap.currentStep = null;
        snap.completedSteps = [];
        break;
      }
      case 'phase_step_start': {
        const ev = event as unknown as Record<string, unknown>;
        snap.currentStep = (ev.step as string) ?? null;
        break;
      }
      case 'phase_step_complete': {
        const ev = event as unknown as Record<string, unknown>;
        snap.completedSteps.push({
          step: (ev.step as string) ?? '',
          durationMs: (ev.durationMs as number) ?? 0,
          costUsd: snap.sessionCostUsd,
        });
        break;
      }
      case 'phase_complete': {
        const ev = event as unknown as Record<string, unknown>;
        snap.completedPhases.push({
          phaseNumber: (ev.phaseNumber as string) ?? snap.currentPhase ?? '',
          phaseName: (ev.phaseName as string) ?? snap.currentPhaseName ?? '',
          success: (ev.success as boolean) ?? true,
        });
        break;
      }
      case 'cost_update': {
        const ev = event as unknown as Record<string, unknown>;
        snap.sessionCostUsd = (ev.sessionCostUsd as number) ?? snap.sessionCostUsd;
        snap.cumulativeCostUsd = (ev.cumulativeCostUsd as number) ?? snap.cumulativeCostUsd;
        break;
      }
      case 'session_complete': {
        snap.status = 'complete';
        const ev = event as unknown as Record<string, unknown>;
        snap.sessionCostUsd = (ev.totalCostUsd as number) ?? snap.sessionCostUsd;
        break;
      }
      case 'session_error': {
        snap.status = 'error';
        const ev = event as unknown as Record<string, unknown>;
        snap.sessionCostUsd = (ev.totalCostUsd as number) ?? snap.sessionCostUsd;
        break;
      }
      case 'milestone_start': {
        // phaseCount available but not in snapshot — intentionally ignored
        break;
      }
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
