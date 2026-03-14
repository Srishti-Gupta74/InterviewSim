
# InterviewSim

InterviewSim is a web-based AI interview simulation app built with React and Vite. It conducts structured interviews across multiple phases, evaluates candidate answers with Gemini, and generates a final AI-based scorecard.

## Core Features

- Full AI interview flow powered by Gemini 2.5 Flash
- Structured interview phases:
	- Introduction
	- Technical
	- Problem Solving
	- Managerial
	- Closing
- Adaptive next-question behavior based on previous answer quality
- Candidate and Interviewer views
- Candidate mode hides live scoring for realistic interview behavior
- Interviewer mode shows live evaluation signals
- Anti-cheat tab switch detection and report logging
- Session persistence to survive accidental tab refresh
- Final report generation and download:
	- JSON export
	- PDF download
- Built-in analytics:
	- Average response time
	- Fastest answer
	- Slowest answer
	- Skill coverage summary
- Light and dark theme toggle with saved preference

## Tech Stack

- React
- Vite
- Gemini API (model: gemini-2.5-flash)
- jsPDF for PDF export

## Project Structure

- src/App.jsx: Main application logic and UI
- src/main.jsx: App entry point
- .github/workflows/deploy-pages.yml: GitHub Pages deployment workflow
- vite.config.js: Vite config (includes envDir mapping)

## Environment Setup

Set your Gemini API key as:

VITE_GEMINI_API_KEY=your_key_here

This project is configured with envDir: .. in vite.config.js, so it can read .env from the parent folder of this project.

You can place .env in either location:

- Parent folder (current setup):
	- Mesra/.env
- Project folder:
	- interview-sim/.env

After changing .env, restart the dev server.

## Local Development

Run from the interview-sim folder:

1. Install dependencies
	 npm install

2. Start development server
	 npm run dev

3. Build production bundle
	 npm run build

4. Lint project
	 npm run lint

## Interview Flow

1. Fill candidate profile and role details
2. Start interview
3. Answer AI-generated questions phase by phase
4. End interview to generate final AI scorecard
5. Download report as PDF or JSON

## API Usage Notes

To reduce quota burn while keeping AI-based final evaluation:

- Questions are shortened to one per phase
- Final evaluation remains Gemini-based
- Keep responses concise but meaningful

If your API quota is exhausted, final scorecard generation can fail until quota resets.

## Deployment (GitHub Pages)

This repository includes a Pages workflow in .github/workflows/deploy-pages.yml.

Deployment behavior:

- Trigger: push to main
- Build command includes repo base path for Pages:
	- npm run build -- --base "/<repo-name>/"

Required GitHub settings:

1. Push this project to a GitHub repository
2. In repository settings, open Pages
3. Set Source to GitHub Actions
4. Push to main and wait for the workflow to complete

Your site will be available at:

https://<your-username>.github.io/<repo-name>/

## Troubleshooting

- npm run dev fails with ENOENT:
	- Ensure you are inside the interview-sim folder when running commands

- App returns to landing page after tab switch:
	- Session restore is enabled; if browser privacy settings block storage, disable strict storage restrictions

- Missing API key error:
	- Verify VITE_GEMINI_API_KEY is defined and dev server is restarted

- Final report generation fails:
	- Check Gemini quota and billing status

## License

Private project. Add your preferred license before public release.
