/**
 * Data Loader Module
 * Handles loading and parsing of JSON data files
 */

class DataLoader {
    constructor() {
        this.slides = [];
        this.notes = [];
        this.exercises = [];
        this.exams = [];
        this.studyGuide = []; // Study guide sections for search
        this.loaded = false;
    }

    async loadAll() {
        try {
            const [slidesData, notesData, exercisesData, examsData, studyGuideData] = await Promise.all([
                this.loadJSON('data/slides.json'),
                this.loadJSON('data/notes.json'),
                this.loadJSON('data/exercises.json'),
                this.loadJSON('data/past_exams.json'),
                this.loadJSON('data/study_guide.json')
            ]);

            this.slides = slidesData.slides || [];
            this.notes = notesData.notes || [];
            this.exercises = exercisesData.exercises || [];
            this.exams = examsData.exams || [];
            this.studyGuide = studyGuideData.studyGuide || [];
            this.loaded = true;

            return {
                slides: this.slides,
                notes: this.notes,
                exercises: this.exercises,
                exams: this.exams,
                studyGuide: this.studyGuide
            };
        } catch (error) {
            console.error('Error loading data:', error);
            // Return empty data if files don't exist yet
            return {
                slides: [],
                notes: [],
                exercises: [],
                exams: [],
                studyGuide: []
            };
        }
    }

    async loadJSON(path) {
        try {
            const response = await fetch(path);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.warn(`Could not load ${path}:`, error);
            return {};
        }
    }

    getSlides() {
        return this.slides;
    }

    getNotes() {
        return this.notes;
    }

    getExercises() {
        return this.exercises;
    }

    getExams() { // Added for past exams
        return this.exams;
    }

    getAllTopics() {
        const topics = new Set();

        this.slides.forEach(slide => {
            if (slide.topic) topics.add(slide.topic);
        });

        this.notes.forEach(note => {
            if (note.topic) topics.add(note.topic);
        });

        this.exercises.forEach(exercise => {
            if (exercise.topic) topics.add(exercise.topic);
        });

        this.exams.forEach(exam => { // Added for past exams
            if (exam.topic) topics.add(exam.topic);
        });

        return Array.from(topics).sort();
    }

    getSlideTopics() {
        const topics = new Set(this.slides.map(s => s.topic).filter(Boolean));
        return Array.from(topics).sort();
    }

    getNoteTopics() {
        const topics = new Set(this.notes.map(n => n.topic).filter(Boolean));
        return Array.from(topics).sort();
    }

    getExerciseTopics() {
        const topics = new Set(this.exercises.map(e => e.topic).filter(Boolean));
        return Array.from(topics).sort();
    }

    getExamTopics() { // Added for past exams
        const topics = new Set(this.exams.map(e => e.topic).filter(Boolean));
        return Array.from(topics).sort();
    }
}

// Global instance
const dataLoader = new DataLoader();
