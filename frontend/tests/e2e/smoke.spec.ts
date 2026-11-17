import { expect, test } from '@playwright/test'
import { stubApi } from './fixtures'

test.describe('Veritas smoke', () => {
  test.beforeEach(async ({ page }) => {
    await stubApi(page)
  })

  test('analyze page explains the server state and offers the modes it can run', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Transparent AI-text')
    await expect(page.getByRole('tab', { name: /Upload/ })).toBeVisible()
    await expect(page.getByRole('button', { name: 'standard' })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'lite' })).toBeEnabled()
    await expect(page.getByText('corpus: 30 documents')).toBeVisible()
  })

  test('paste text and reach a results page with the verdict shown in the DOM', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('tab', { name: /Paste/ }).click()
    await page.getByRole('textbox').first().fill('Alpha beta gamma delta. Epsilon zeta eta theta.')
    await page.getByRole('button', { name: 'Analyse', exact: true }).click()
    await page.waitForURL('**/results/test-analysis')
    await expect(page.getByTestId('document-viewer')).toContainText('Alpha beta gamma delta.')
    await expect(page.getByTestId('verdict-text')).toHaveText('71.0%')
    // The unavailable component is named rather than hidden.
    await expect(page.getByText('binoculars unavailable')).toBeVisible()
    // Real corpus size, not a marketing number.
    await expect(page.getByText('corpus 30')).toBeVisible()
  })

  test('results page works with effects off and no WebGL context', async ({ page }) => {
    await page.addInitScript(() => {
      const original = HTMLCanvasElement.prototype.getContext
      HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, type: string, ...rest: unknown[]) {
        if (type === 'webgl' || type === 'webgl2') return null
        return (original as never as (...a: unknown[]) => unknown).call(this, type, ...rest)
      } as never
    })
    await page.goto('/results/test-analysis')
    await expect(page.getByTestId('verdict-text')).toHaveText('71.0%')
    await expect(page.locator('canvas')).toHaveCount(0)
    await expect(page.getByRole('img', { name: /Probability of machine authorship/ }).first()).toBeVisible()
    await expect(page.getByTestId('document-viewer')).toContainText('Epsilon zeta eta theta.')
  })

  test('evaluation page puts held-out results before in-domain results', async ({ page }) => {
    await page.goto('/evaluation')
    const headings = await page.locator('h2').allInnerTexts()
    const ood = headings.findIndex((text) => text.includes('Held-out'))
    const inDomain = headings.findIndex((text) => text.includes('In-domain'))
    expect(ood).toBeGreaterThanOrEqual(0)
    expect(inDomain).toBeGreaterThan(ood)
    await expect(page.getByText('SMOKE RUN — not results')).toHaveCount(0)
    await expect(page.getByText('third-party, OOD only')).toBeVisible()
    await expect(page.getByText('synthetic')).toBeVisible()
  })

  test('history lists an analysis and opens it', async ({ page }) => {
    await page.goto('/history')
    await expect(page.getByRole('cell', { name: 'sample.txt' })).toBeVisible()
    await page.getByRole('link', { name: 'Open' }).first().click()
    await page.waitForURL('**/results/test-analysis')
    await expect(page.getByTestId('verdict-text')).toBeVisible()
  })

  test('about page states the methodology and the limitations', async ({ page }) => {
    await page.goto('/about')
    await expect(page.getByRole('heading', { name: 'About Veritas' })).toBeVisible()
    await expect(page.getByText(/never base an accusation/)).toBeVisible()
    await expect(page.getByText('Plagiarism corpus on this server: 30 documents')).toBeVisible()
  })

  test('unknown route renders the custom 404', async ({ page }) => {
    await page.goto('/nope')
    await expect(page.getByText('404')).toBeVisible()
  })
})
