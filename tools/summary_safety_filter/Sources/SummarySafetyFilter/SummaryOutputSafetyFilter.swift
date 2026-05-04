import Foundation

nonisolated enum SummaryOutputSafetyRisk: String, Codable, Sendable, Equatable {
    case promptLeakage
    case ungroundedHighRiskTerm
    case selfHarmInstruction
    case hateOrHarassment
    case sexualContent
    case sexualMinor
    case violentInstruction
    case weaponsInstruction
    case cyberAbuseInstruction
    case illicitDrugInstruction
    case unboundedOutput
}

nonisolated enum SummaryOutputSafetyDecision: Equatable, Sendable {
    case allowed(String)
    case suppressed(SummaryOutputSafetyRisk)
}

nonisolated struct SummaryOutputSafetyFilter: Sendable {
    static let version = "risk-v1"

    private enum RiskGroup: CaseIterable {
        case selfHarm
        case hate
        case sexual
        case violence
        case weapons
        case cyber
        case illicitDrugs

        var suppressionReason: SummaryOutputSafetyRisk {
            switch self {
            case .selfHarm:
                return .selfHarmInstruction
            case .hate:
                return .hateOrHarassment
            case .sexual:
                return .sexualContent
            case .violence:
                return .violentInstruction
            case .weapons:
                return .weaponsInstruction
            case .cyber:
                return .cyberAbuseInstruction
            case .illicitDrugs:
                return .illicitDrugInstruction
            }
        }
    }

    private static let maximumDisplayCharacters = 280
    private static let maximumDisplayWords = 45

    private static let promptLeakagePhrases = [
        "abstract below",
        "as an ai",
        "developer message",
        "hidden policies",
        "hidden instruction",
        "ignore previous",
        "ignore the previous",
        "ignore the summary length limit",
        "malicious user",
        "paper summary model",
        "reveal the prompt",
        "system prompt",
        "unsafe test output",
        "unsafe topics"
    ]

    private static let actionPhrases = [
        "attack",
        "build",
        "bypass",
        "evade",
        "exploit",
        "exfiltrate",
        "guide",
        "guide to",
        "how to",
        "instructions",
        "make",
        "manufacture",
        "payload",
        "recipe",
        "steps",
        "steps to",
        "steal",
        "synthesize",
        "tutorial",
        "undetectable",
        "weaponize"
    ]

    private static let sexualMinorPhrases = [
        "child sexual",
        "csam",
        "minor sexual",
        "underage sexual"
    ]

    private static let directSelfHarmPhrases = [
        "guide to self harm",
        "guide to self harming",
        "how to self harm",
        "self harm methods",
        "steps to self harm",
        "suicide methods"
    ]

    private static let hateOrHarassmentPhrases = [
        "ethnic cleansing",
        "racial purity",
        "racial superiority"
    ]

    private static let explicitSexualPhrases = [
        "explicit sex",
        "graphic sexual",
        "pornographic"
    ]

    private static let riskTermsByGroup: [RiskGroup: [String]] = [
        .selfHarm: [
            "cutting",
            "overdose",
            "self harm",
            "self harming",
            "suicide"
        ],
        .hate: [
            "antisemitic",
            "hate speech",
            "racist",
            "slur"
        ],
        .sexual: [
            "nude",
            "porn",
            "rape",
            "sexual"
        ],
        .violence: [
            "assassination",
            "beheading",
            "kill",
            "massacre",
            "murder",
            "torture"
        ],
        .weapons: [
            "bioweapon",
            "bomb",
            "explosive",
            "firearm",
            "nerve agent",
            "poison",
            "toxin",
            "weapon"
        ],
        .cyber: [
            "backdoor",
            "botnet",
            "credential",
            "ddos",
            "exploit",
            "keylogger",
            "malware",
            "payload",
            "phishing",
            "ransomware",
            "rootkit"
        ],
        .illicitDrugs: [
            "cocaine",
            "fentanyl",
            "heroin",
            "lsd",
            "meth"
        ]
    ]

    func evaluate(output: String, for paper: Paper) -> SummaryOutputSafetyDecision {
        let displayText = collapseWhitespace(in: output)
        guard !displayText.isEmpty else {
            return .allowed(displayText)
        }

        if displayText.count > Self.maximumDisplayCharacters || wordCount(in: displayText) > Self.maximumDisplayWords {
            return .suppressed(.unboundedOutput)
        }

        let normalizedOutput = normalizedForMatching(displayText)
        if containsAny(Self.promptLeakagePhrases, in: normalizedOutput) {
            return .suppressed(.promptLeakage)
        }
        if containsAny(Self.sexualMinorPhrases, in: normalizedOutput) {
            return .suppressed(.sexualMinor)
        }
        if containsAny(Self.directSelfHarmPhrases, in: normalizedOutput) {
            return .suppressed(.selfHarmInstruction)
        }
        if containsAny(Self.hateOrHarassmentPhrases, in: normalizedOutput) {
            return .suppressed(.hateOrHarassment)
        }
        if containsAny(Self.explicitSexualPhrases, in: normalizedOutput) {
            return .suppressed(.sexualContent)
        }

        let normalizedSource = normalizedForMatching("\(paper.title) \(paper.abstract)")
        let outputGroups = matchingRiskGroups(in: normalizedOutput)
        let sourceGroups = matchingRiskGroups(in: normalizedSource)
        if outputGroups.contains(where: { !sourceGroups.contains($0) }) {
            return .suppressed(.ungroundedHighRiskTerm)
        }

        if containsAny(Self.actionPhrases, in: normalizedOutput),
           let reason = outputGroups.map(\.suppressionReason).first {
            return .suppressed(reason)
        }

        return .allowed(displayText)
    }

    private func matchingRiskGroups(in normalizedText: String) -> Set<RiskGroup> {
        var groups: Set<RiskGroup> = []
        for group in RiskGroup.allCases {
            guard let terms = Self.riskTermsByGroup[group] else { continue }
            if containsAny(terms, in: normalizedText) {
                groups.insert(group)
            }
        }
        return groups
    }

    private func containsAny(_ phrases: [String], in normalizedText: String) -> Bool {
        phrases.contains { phrase in
            containsPhrase(phrase, in: normalizedText)
        }
    }

    private func containsPhrase(_ phrase: String, in normalizedText: String) -> Bool {
        let normalizedPhrase = normalizedForMatching(phrase)
        guard !normalizedPhrase.isEmpty else { return false }
        return " \(normalizedText) ".contains(" \(normalizedPhrase) ")
    }

    private func collapseWhitespace(in text: String) -> String {
        text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func normalizedForMatching(_ text: String) -> String {
        let folded = text
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
        let scalars = folded.unicodeScalars.map { scalar -> Character in
            CharacterSet.alphanumerics.contains(scalar) ? Character(scalar) : " "
        }
        return collapseWhitespace(in: String(scalars))
    }

    private func wordCount(in text: String) -> Int {
        collapseWhitespace(in: text)
            .split(separator: " ")
            .count
    }
}
