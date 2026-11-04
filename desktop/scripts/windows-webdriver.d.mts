export function windowsAttachment(env: NodeJS.ProcessEnv): { debuggerAddress: string; port: number };

export interface WindowsBuildBinding {
  debugPort: number;
  identifier: string;
}

export class WindowsWebDriver {
  constructor(options: {
    application: string;
    driver: string;
    cwd: string;
    env: NodeJS.ProcessEnv;
  });
  prepare(options: {
    baseConfig: { app: { windows: Array<Record<string, unknown>> } };
    e2eConfig: { identifier: string; app?: Record<string, unknown> };
    wryVersion: string;
  }, previous?: WindowsBuildBinding): Promise<{
    binding: WindowsBuildBinding;
    configuration: { identifier: string; app: { windows: Array<Record<string, unknown>> } };
  }>;
  start(): Promise<{ CORPI_E2E_DEBUGGER_ADDRESS: string; CORPI_E2E_DRIVER_PORT: string }>;
  stop(): Promise<void>;
}
