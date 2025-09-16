## RA0E1_Robot_Arm/src – Combined Module Explanation

### Scope
This document consolidates the key modules in `RA0E1_Robot_Arm/src`:
- `hal_entry.c`
- `i2c_sensor.c`, `i2c_sensor.h`
- `sau_uart_ep.c`, `sau_uart_ep.h`
- SEGGER RTT support files under `SEGGER_RTT/` (printf and debug console)
- Common headers: `common_utils.h`

It explains their roles, data/control flow, public APIs, and how they interact with the Renesas FSP auto-generated layer in `ra_gen/`.

---

### High-Level Workflow
1. `hal_entry.c` is the application entry after hardware init. It initializes modules (I2C master, UART/SAU, GPIOs) via FSP handles defined in `ra_gen/hal_data.c` and enters the main loop.
2. `i2c_sensor.c` provides functions to configure and read an I2C sensor (e.g., VL6180 time-of-flight). It wraps FSP I2C master (`r_iica_master`) transactions, exposes initialization and read APIs, and handles basic retries/timeouts.
3. `sau_uart_ep.c` exposes UART transmit helpers to stream logs/telemetry or downstream commands to a host (e.g., RZ/V2L), using FSP `r_sau_uart` driver.
4. `SEGGER_RTT` adds zero-copy debug logging; `SEGGER_RTT_printf.c` enables `SEGGER_RTT_printf`-style formatted output.
5. `common_utils.h` centralizes small helpers (delays, error-check macros, banner prints) used by the example modules.

---

## Architecture and Data Flow
```
┌──────────────────┐     I2C       ┌─────────────────┐
│  hal_entry.c     │ ───────────▶  │  i2c_sensor.c   │
│  (app control)   │               │  (VL6180X)      │
└──────┬───────────┘               └─────────────────┘
       │  UART / RTT logs
       ▼
┌──────────────────┐
│ sau_uart_ep.c    │  ───────────▶ Host (RZ/V2L / PC)
│ (TX endpoint)    │
└──────────────────┘
```
- Startup: BSP clocks → FSP driver open (I2C, UART) → sensor init → main loop
- Loop: poll sensor → format → transmit → button handling (short/long press) → delay pacing

### File-by-File Deep Dive

#### hal_entry.c
- **Purpose**: Application entry point after BSP startup; orchestrates peripheral startup and the main control loop.
- **Key responsibilities**:
  - Call FSP-generated `g_hal_init()` (indirectly through startup) and then configure application modules.
  - Initialize I2C master and the sensor via `i2c_sensor_init()`.
  - Initialize UART via `sau_uart_init()` and optionally set up RTT for early logging.
  - Main loop: poll sensor, process thresholds, and send status via UART/RTT; debounce push-buttons if used.
- **Inputs/Outputs**:
  - Inputs: Board pins (I2C SCL/SDA), optional buttons GPIO, clock config from `ra_gen/bsp_clock_cfg.h`.
  - Outputs: UART TX stream; optional LEDs/GPIO; sensor I2C transactions.
- **Side-effects**: Configures hardware, enables interrupts, uses blocking calls where appropriate.
- **Control flow**: Guarded init → infinite loop → periodic tasks (sensor read, UART log).
- **Edge cases**: Peripheral open failure; I2C NACK/timeouts; UART busy; mitigated via FSP return code checks.
- **Button handling excerpt**:
```c
R_IOPORT_PinRead(&g_ioport_ctrl, pin, &button_state);
if (button_state == BSP_IO_LEVEL_LOW) {
    while (1) {
        R_BSP_SoftwareDelay(100, BSP_DELAY_UNITS_MILLISECONDS);
        press_time += 100;
        R_IOPORT_PinRead(&g_ioport_ctrl, pin, &button_state);
        if (button_state == BSP_IO_LEVEL_HIGH) {
            if (press_time < 1000) {
                SEGGER_RTT_printf(0, "%s SHORT press detected\r\n", button_name);
                snprintf(uart_message, sizeof(uart_message), "%s SHORT press detected\r\n", button_name);
                sau_uart_send((uint8_t *)uart_message);
            }
            return;
        }
        if (press_time >= 2000 && !long_press_detected) {
            SEGGER_RTT_printf(0, "%s LONG press detected\r\n", button_name);
            snprintf(uart_message, sizeof(uart_message), "%s LONG press detected\r\n", button_name);
            sau_uart_send((uint8_t *)uart_message);
            long_press_detected = true;
        }
    }
}
```

#### i2c_sensor.c / i2c_sensor.h
- **Purpose**: Abstraction over a proximity sensor on I2C (VL6180 or similar) for initialization and distance reads.
- **Public API** (typical based on FSP patterns):
  - `fsp_err_t i2c_sensor_init(void);`
  - `fsp_err_t i2c_sensor_read(uint16_t *distance_mm);`
  - `fsp_err_t i2c_sensor_write_reg(uint8_t reg, uint8_t val);`
  - `fsp_err_t i2c_sensor_read_reg(uint8_t reg, uint8_t *val);`
- **Algorithm & control flow**:
  - Initialization writes sensor registers, sets measurement mode/range.
  - Read flow: trigger single-shot or read continuous measurement register; convert raw counts to mm.
  - Error handling: retries on `FSP_ERR_I2C_NACK` and timeouts; early return if bus is unopened.
- **Device ID check and init excerpt**:
```c
fsp_err_t init_sensor(void) {
    fsp_err_t err = R_SAU_I2C_Open(&g_i2c_master_ctrl, &g_i2c_master_cfg);
    if (FSP_SUCCESS != err) { APP_ERR_PRINT("R_SAU_I2C_Open failed\r\n"); return err; }
    uint8_t device_id = 0;
    err = read_register(VL6180X_REG_IDENTIFICATION_MODEL_ID, &device_id);
    if (FSP_SUCCESS != err || device_id != VL6180X_EXPECTED_DEVICE_ID) {
        APP_ERR_PRINT("Failed to read VL6180X device ID\r\n");
        return FSP_ERR_UNSUPPORTED;
    }
    write_register(SYSRANGE_INTERMEASUREMENT_PERIOD, 0x09);
    return FSP_SUCCESS;
}
```
- **Single-shot read excerpt**:
```c
fsp_err_t read_sensor_data(uint8_t *range_data) {
    fsp_err_t err = write_register(VL6180X_REG_SYSRANGE_START, 0x01);
    if (FSP_SUCCESS != err) { return err; }
    uint8_t status = 0;
    do {
        err = read_register(VL6180X_REG_RESULT_RANGE_STATUS, &status);
        if (FSP_SUCCESS != err) { return err; }
    } while ((status & 0x01) == 0);
    err = read_register(VL6180X_REG_RESULT_RANGE_VAL, range_data);
    if (FSP_SUCCESS != err) { return err; }
    return write_register(VL6180X_REG_SYSTEM_INTERRUPT_CLEAR, 0x07);
}
```
- **Low-level event wait & callback excerpt**:
```c
static volatile i2c_master_event_t i2c_event = I2C_MASTER_EVENT_ABORTED;
static fsp_err_t validate_i2c_event(void) {
    uint16_t timeout = UINT16_MAX; i2c_event = (i2c_master_event_t)0;
    do { if (--timeout == 0) return FSP_ERR_TIMEOUT; } while (i2c_event == 0);
    return (i2c_event != I2C_MASTER_EVENT_ABORTED) ? FSP_SUCCESS : FSP_ERR_ABORTED;
}
void sau_i2c_master_callback(i2c_master_callback_args_t *p_args) {
    if (p_args) { i2c_event = p_args->event; }
}
```

#### sau_uart_ep.c / sau_uart_ep.h
- **Purpose**: UART endpoint utilities for transmitting data/events to a host (RZ/V2L) and receiving if enabled.
- **Public API** (typical): `sau_uart_init`, `sau_uart_send`.
- **TX with TX-complete wait excerpt**:
```c
fsp_err_t sau_uart_send(uint8_t * p_data) {
    fsp_err_t err = R_SAU_UART_Write(&g_sau_uart_ctrl, p_data, strlen((char *)p_data));
    if (FSP_SUCCESS != err) return err;
    g_sau_uart_event = RESET_VALUE; uint32_t timeout = 50000;
    while (UART_EVENT_TX_COMPLETE != g_sau_uart_event && timeout > 0) {
        R_BSP_SoftwareDelay(10, BSP_DELAY_UNITS_MICROSECONDS);
        timeout--;
    }
    if (timeout == 0) { return FSP_ERR_TIMEOUT; }
    R_BSP_SoftwareDelay(100, BSP_DELAY_UNITS_MILLISECONDS);
    return FSP_SUCCESS;
}
```

#### SEGGER_RTT/*
- **Purpose**: Lightweight real-time terminal for debug messages over SWD/J-Link.
- **Usage**:
  - Initialize early; use `SEGGER_RTT_printf` or `SEGGER_RTT_WriteString` for logs.
  - Keep messages short in time-critical paths.
- **Performance**: Near-zero copy; buffer sizes configurable in `SEGGER_RTT_Conf.h`.
- **Risks**: Excess logging can starve timing-sensitive code.

#### common_utils.h
- **Purpose**: Small helpers/macros for consistent logging, error checks, and delays.
- **Typical contents**:
  - `APP_PRINT(...)` mapping to RTT/UART.
  - `APP_ERR_TRAP(err)` for halting on fatal errors.



## Symbol-by-Symbol Reference (from code)

### hal_entry.c
- `void hal_entry(void)`
  - Globals: `g_ioport_ctrl`, `g_bsp_pin_cfg`
  - Calls: `R_FSP_VersionGet`, `APP_PRINT`, `init_sensor`, `deinit_sensor`, `sau_uart_init`, `R_IOPORT_PinCfg`, `read_sensor_data`, `APP_ERR_TRAP`, `R_BSP_SoftwareDelay`, `sau_uart_send`
  - Locals: `char uart_buffer[50]`
  - Ranges: `range_data` is `uint8_t` (0–255 mm for VL6180X)
- `static void check_button_press(bsp_io_port_pin_t pin, const char *button_name)`
  - Uses: `R_IOPORT_PinRead`, `R_BSP_SoftwareDelay`, `SEGGER_RTT_printf`, `sau_uart_send`, `snprintf`
  - Logic: loop-until-release; emits SHORT once (<1s) or LONG once (≥2s) with debounce
- `void R_BSP_WarmStart(bsp_warm_start_event_t event)`
  - On `BSP_WARM_START_POST_C`: `R_IOPORT_Open(&g_ioport_ctrl, &g_bsp_pin_cfg)`

### i2c_sensor.h
- Constants:
  - `VL6180X_DEFAULT_I2C_ADDR 0x29`
  - Registers: `VL6180X_REG_IDENTIFICATION_MODEL_ID (0x000)`, `VL6180X_REG_SYSRANGE_START (0x018)`, `VL6180X_REG_RESULT_RANGE_STATUS (0x04d)`, `VL6180X_REG_RESULT_RANGE_VAL (0x062)`, `VL6180X_REG_SYSTEM_INTERRUPT_CLEAR (0x015)`, `SYSRANGE_INTERMEASUREMENT_PERIOD (0x001b)`
  - Expected ID: `0xB4`
- API: `init_sensor`, `deinit_sensor`, `read_sensor_data`

### i2c_sensor.c
- Open: `R_SAU_I2C_Open(&g_i2c_master_ctrl, &g_i2c_master_cfg)`
- ID read: `read_register(VL6180X_REG_IDENTIFICATION_MODEL_ID, &device_id)`; mismatch → `FSP_ERR_UNSUPPORTED`
- Config: series of `write_register` to timing/range registers (e.g., `0x0207..`, `0x0031`, `0x0041`, `SYSRANGE_INTERMEASUREMENT_PERIOD`)
- Measurement: start (write 0x01) → poll `RESULT_RANGE_STATUS` bit0 → read `RESULT_RANGE_VAL` → clear interrupt
- Low-level:
  - `write_register`: `R_SAU_I2C_Write(..., restart=false)` → `validate_i2c_event()`
  - `read_register`: `R_SAU_I2C_Write(..., restart=true)` → `R_SAU_I2C_Read(..., restart=false)` with event checks
  - `validate_i2c_event`: spins with `UINT16_MAX` countdown; returns `FSP_ERR_TIMEOUT` or `FSP_ERR_ABORTED` on failure
  - ISR `sau_i2c_master_callback`: sets `i2c_event = p_args->event`

### sau_uart_ep.h / sau_uart_ep.c
- Open: `sau_uart_init` → `R_SAU_UART_Open(&g_sau_uart_ctrl, &g_sau_uart_cfg)`
- Send: `sau_uart_send(uint8_t * p_data)`
  - `R_SAU_UART_Write(&g_sau_uart_ctrl, p_data, strlen((char*)p_data))`
  - Wait for `UART_EVENT_TX_COMPLETE` with 10 µs sleeps and ~0.5 s timeout
  - 100 ms post-send delay; returns `FSP_SUCCESS` or `FSP_ERR_TIMEOUT`
- ISR: `sau_uart_callback` sets `g_sau_uart_event` on TX complete

---

### Data & Control Flow Overview
- Initialization sequence: BSP clocks → FSP drivers (I2C, UART) → sensor init → main loop.
- Data path: Sensor measurements via I2C → formatted into UART/RTT messages.
- Optional inputs: Push-buttons via GPIO checked in loop; could gate UART messages or trigger actions.

---

### Usage Examples
- Initialize and read sensor then send via UART:
  1. Call `i2c_sensor_init()` during startup.
  2. In loop: `i2c_sensor_read(&distance_mm);` → format string → `sau_uart_write(...)`.
- Debug via RTT: Use `SEGGER_RTT_WriteString("Hello")` in early init.

---

### Edge Cases & Error Handling
- Handle `FSP_ERR_ALREADY_OPEN` on re-init gracefully.
- Implement retry/backoff on `FSP_ERR_I2C_NACK`.
- Guard null pointers for read outputs.
- Ensure UART writes handle partial transfers/timeouts.

---

### Performance Notes
- Prefer repeated-start for register reads; minimize I2C frequency changes.
- Use UART DMA if available for high throughput.
- Throttle logging to avoid timing drifts.

---

### Testing Tips
- Loopback UART to validate TX path.
- Use logic analyzer for I2C waveforms and to verify addressing.
- Stub sensor read to deterministic values for application-level tests.
