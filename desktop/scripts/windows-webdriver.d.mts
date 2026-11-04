export function windowsAttachment(env: NodeJS.ProcessEnv): { debuggerAddress: string; port: number };

export class WindowsWebDriver {
  constructor(options: {
    application: string;
    driver: string;
    cwd: string;
    env: NodeJS.ProcessEnv;
    scratch: string;
  });
  start(): Promise<{ CORPI_E2E_DEBUGGER_ADDRESS: string; CORPI_E2E_DRIVER_PORT: string }>;
  stop(): Promise<void>;
}
