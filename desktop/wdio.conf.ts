import type { Options } from "@wdio/types";

const APPLICATION = "./src-tauri/target/release/corpi-desktop.exe";

export const config: Options.Testrunner = {
  specs: ["./e2e/specs/**/*.ts"],
  runner: "local",
  maxInstances: 1,
  logLevel: "warn",
  framework: "mocha",
  reporters: ["spec"],
  waitforTimeout: 15000,
  connectionRetryCount: 3,
  connectionRetryTimeout: 30000,
  hostname: "127.0.0.1",
  port: 4444,
  capabilities: [
    {
      browserName: "wry",
      "tauri:options": {
        application: APPLICATION,
      },
    } as WebdriverIO.Capabilities,
  ],
};
