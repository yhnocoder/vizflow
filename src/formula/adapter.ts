export interface FormulaAdapter {
  render(tex: string, display: boolean, target: HTMLElement): void | Promise<void>;
}
